#include <memory>
#include <cassert>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h"
#include "triton/Analysis/TXLUtility.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>
#include <memory>

using namespace mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttxg = mlir::triton::txlgpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir::triton::txlgpu {

#define GEN_PASS_DEF_TXLGPUWSCODEPARTITION
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-ws-code-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")


/***********************************
 * Async Task Specialization
***********************************/

// Returns arrA - arrB (elements in arrA not in arrB)
// Also verifies that arrB is a subset of arrA
void subtractArrayRefs(llvm::ArrayRef<int32_t> arrA, 
                       llvm::ArrayRef<int32_t> arrB,
                       SmallVector<int32_t> &result) {
  // First verify arrB is a subset of arrA
  for (const auto &elem : arrB) {
    if (!llvm::is_contained(arrA, elem)) {
      llvm::report_fatal_error("arrB is not a subset of arrA!");
    }
  }

  // Copy elements from arrA that aren't in arrB
  for (const auto &elem : arrA) {
    if (!llvm::is_contained(arrB, elem)) {
      result.push_back(elem);
    }
  }
}

std::pair<int, bool> scanRegUsage(Block *block, int asyncTaskId,
                                  int regDecProducer, int regIncConsumer,
                                  int numWarpgroups, bool hasTma) {
  // TODO: scan ops to estimate register usage
  // TODO: based on cutlass
  if (asyncTaskId == 0) {
    // deallocate registers
    if (numWarpgroups == 2)
        regDecProducer = 56;
    else if (numWarpgroups == 3)
        //regDecProducer = hasTma ? 24 : 40;
        regDecProducer = 40;
    else
        regDecProducer = 32;

    return {regDecProducer, false};
  } else {
    // allocate registers
    if (numWarpgroups == 2)
        regIncConsumer = 256;
    else if (numWarpgroups == 3)
        //regIncConsumer = hasTma ? 240 : 232;
        regIncConsumer = 232;
    else
        regIncConsumer = 160;

    return {regIncConsumer, true};
  }
}

bool hasWarpgroupOperand(Value value) {
  // Base case: this value is defined by a warpgroup op
  if (auto op = value.getDefiningOp()) {
    if (isa<tt::IsWarpgroupOp>(op)) {
      return true;
    }
  }

  // For operation results, check all operands of the defining op
  if (auto op = value.getDefiningOp()) {
    for (Value operand : op->getOperands()) {
      if (hasWarpgroupOperand(operand)) {
        return true;
      }
    }
  }

  return false;
}

bool isWarpgroupIf(Operation* op) {
    if (isa<scf::IfOp>(op)){
        scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
        Value cond = ifOp.getCondition();
        if (auto defOp = cond.getDefiningOp()) {
          if (isa<tt::IsWarpgroupOp>(defOp)) {
            return true;
          }
          if (hasWarpgroupOperand(cond))
              op->emitError("is_warpgroup should always be used directly in if condition!\n");
          //assert(!hasWarpgroupOperand(cond) && "is_warpgroup should always be used directly in if condition!\n");
        }
    }
    return false;
}

// Find the outermost if with tt::IsWarpgroupOp condition
SmallVector<scf::IfOp> findOuterWarpGroupIf(Operation* op) {
  SmallVector<scf::IfOp> wgOps;

  // Recursively check nested operations
  for (Region& region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation& op : block) {
        if (isWarpgroupIf(&op)) {
          wgOps.push_back(dyn_cast<scf::IfOp>(op));
        }
      }
    }
  }
  return wgOps;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilder &builder,
                        int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds);

// wargroup if
Operation *SpecializeWarpgroupIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilder &builder,
                          int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds) {
  Value cond = ifOp.getCondition();
  assert(isa<tt::IsWarpgroupOp>(cond.getDefiningOp()) && "not a isWarpgroup If\n");


  auto warpgroupOp = dyn_cast<tt::IsWarpgroupOp>(cond.getDefiningOp());
  auto ids = warpgroupOp.getIds();
  bool shouldInlineThen = llvm::is_contained(ids, asyncTaskId);
  llvm::ArrayRef<int32_t> selectedTaskIds;
  SmallVector<int32_t> elseIds;
  subtractArrayRefs(rootTaskIds, ids, elseIds);


  Block* selectedBlock;
  Operation* selectedYieldOp;
  if (shouldInlineThen) {
      if (ifOp.thenBlock()){
          selectedBlock = ifOp.thenBlock();
          selectedYieldOp = ifOp.thenYield();
          selectedTaskIds = ids;
      } else {
          return nullptr;
      }
  }
  else {
      if (ifOp.elseBlock()) {
          selectedBlock = ifOp.elseBlock();
          selectedYieldOp = ifOp.elseYield();
          selectedTaskIds = elseIds;
      } else {
          return nullptr;
      }
  }

  LLVM_DEBUG({
    LDBG("specialize Inner Warpgroup ifOp ");

    LDBG("parent tasks");
    for (auto i : rootTaskIds)
        llvm::dbgs() << i << ", ";
    llvm::dbgs() << "\n";
    LDBG("then tasks");
    for (auto i : ids)
        llvm::dbgs() << i << ", ";
    llvm::dbgs() << "\n";
    LDBG("else tasks");
    for (auto i : elseIds)
        llvm::dbgs() << i << ", ";
    llvm::dbgs() << "\n";

    LDBG("selected tasks");
    LDBG(asyncTaskId);

    LDBG("new parent tasks");
    for (auto i : selectedTaskIds)
        llvm::dbgs() << i << ", ";
    llvm::dbgs() << "\n";
    LDBG("Full if");
    ifOp.dump();
    llvm::dbgs() << "\n";
    LDBG("selected block");
    selectedBlock->dump();
    llvm::dbgs() << "\n";
  });


  unsigned resultIdx = 0;
  SmallVector<unsigned> keptResultVec;
  if (!ifOp->getResultTypes().empty()) {
    for (Value yieldV : selectedYieldOp->getOperands()) {
      keptResultVec.push_back(resultIdx);
      ++resultIdx;
    }
  }

  SmallVector<Type> newResultTypes;
  for (auto idx : keptResultVec) {
    newResultTypes.push_back(ifOp->getResultTypes()[idx]);
  }


  // Handle thenRegion of this IfOp.
  for (Operation &selectedOp : selectedBlock->without_terminator()) {
    SpecializeOp(&selectedOp, mapping, builder, asyncTaskId, selectedTaskIds);
  }
  // for other task ids
  //if (ifOp.elseBlock()) {
  //  ifBuilder.setInsertionPoint(ifOp);
  //  for (Operation &elseOp : ifOp.elseBlock()->without_terminator()) {
  //    SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId);
  //  }
  //}

  for (auto idx : keptResultVec) {
    auto oldIfResultSrc = selectedYieldOp->getOperands()[idx]; // then or else
    auto flatIfResult = mapping.lookupOrDefault(oldIfResultSrc);
    assert(flatIfResult && "Unexpected missing mapping");

    mapping.map(ifOp.getResult(idx), flatIfResult);
  }
  return nullptr;
}

// ordinary if
Operation *SpecializeIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilder &builder,
                          int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds) {
  LLVM_DEBUG({
    LDBG("specialize ifOp ");
    ifOp.dump();
  });

  // if op don't have block args, but just yield
  // however, the yield may contain blockarg from parent forop
  unsigned resultIdx = 0;
  SmallVector<unsigned> keptResultVec;
  if (!ifOp->getResultTypes().empty()) {
    for (Value yieldV : ifOp.thenYield().getOperands()) {
      keptResultVec.push_back(resultIdx);
      ++resultIdx;
    }
  }

  SmallVector<Type> newResultTypes;
  for (auto idx : keptResultVec) {
    newResultTypes.push_back(ifOp->getResultTypes()[idx]);
  }

  // Mind the builder insertion point
  // TODO: elseblock direct copy?
  auto newIfOp = builder.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, mapping.lookup(ifOp.getCondition()), true,
      ifOp.elseBlock());
  setOpAttrWgId(newIfOp, asyncTaskId);

  OpBuilder ifBuilder(ifOp.getContext());

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, asyncTaskId, rootTaskIds);
  }
  // Similarly: Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId, rootTaskIds);
    }
  }

  unsigned newResIdx = 0;
  for (auto idx : keptResultVec) {
    mapping.map(ifOp.getResult(idx), newIfOp.getResult(newResIdx));
    ++newResIdx;
  }
  return newIfOp;
}

Operation *SpecializeForOp(scf::ForOp forOp, IRMapping &mapping,
                           OpBuilder &builder,
                           int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds) {

  // Prepare newLoopArgs.
  SmallVector<Value> newLoopArgs;
  for (auto arg : forOp.getInitArgs()) {
    auto newArg = mapping.lookupOrDefault(arg);
    assert(newArg && "Unexpected missing mapping");
    newLoopArgs.push_back(newArg);
  }

  // Prepare loop bounds.
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  // Create newForOp.
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), newLowerBound, newUpperBound, newStep, newLoopArgs);
  setOpAttrWgId(newForOp, asyncTaskId);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

  // Initialize Value mapping from forOp to newForOp
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  // both initargs and region iter args ignore induction vals
  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    auto oldArg = forOp.getRegionIterArgs()[i];
    auto newArg = newForOp.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  // Recursively clone all operations with this asyncTaskId to newForOp.
  OpBuilder forBuilder(forOp.getContext());
  forBuilder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    SpecializeOp(&op, mapping, forBuilder, asyncTaskId, rootTaskIds);
  }

  // Create YieldOp for newForOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (unsigned i =0; i < forOp.getInitArgs().size(); ++i)
    newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));

  bool createNewYield = true;
  if (newForOp.getBody()->mightHaveTerminator()) {
    auto initialYield =
        llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    if (newYieldOperands.size() == 0) {
      createNewYield = false;
    }
  }
  if (createNewYield) {
    auto newYieldOp =
        forBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setOpAttrWgId(newYieldOp, asyncTaskId);
  }

  // Replace results of forOp with results of newForOp.
  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    auto oldResult = forOp.getResult(i);
    auto newResult = newForOp.getResult(i);
    mapping.map(oldResult, newResult);
  }

  return newForOp;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilder &builder,
                        int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds) {
  if (op->getNumRegions() == 0) {
    // case 1: direct clone
    Operation *newOp = builder.clone(*op, mapping);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    setOpAttrWgId(newOp, asyncTaskId);
    return newOp;
  } else {
    // case 2: ops with blocks
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (isWarpgroupIf(ifOp)) {
          return SpecializeWarpgroupIfOp(ifOp, mapping, builder, asyncTaskId, rootTaskIds);
      } else {
          return SpecializeIfOp(ifOp, mapping, builder, asyncTaskId, rootTaskIds);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskId, rootTaskIds);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      setOpAttrWgId(newOp, asyncTaskId);
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

void SpecializeOuterWarpgroupIf(scf::IfOp ifOp, int32_t numWarpgroups, bool hasTma) {

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG("Start specializing region");
  });

  MLIRContext *context = ifOp.getContext();
  OpBuilder builder(ifOp);
  auto loc = ifOp.getLoc();

  Value curAsyncTaskId = builder.create<ttxg::CanonicalWarpgroupIdOp>(loc, builder.getI32Type());
  //Value curAsyncTaskId = builder.create<ttng::GetAsyncTaskIdOp>(loc);

  Value cond = ifOp.getCondition();
  assert(isa<tt::IsWarpgroupOp>(cond.getDefiningOp()) && "not a isWarpgroup If\n");

  auto warpgroupOp = dyn_cast<tt::IsWarpgroupOp>(cond.getDefiningOp());
  auto ids = warpgroupOp.getIds();

  SmallVector<int32_t> rootTaskIds;
  for (int32_t i=0; i < numWarpgroups; i++)
      rootTaskIds.push_back(i);
  SmallVector<int32_t> elseIds;
  subtractArrayRefs(rootTaskIds, ids, elseIds);

  SmallVector<llvm::ArrayRef<int32_t>> idsVec;
  SmallVector<Block*> blockVec;
  idsVec.push_back(ids);
  blockVec.push_back(ifOp.thenBlock());

  if (ifOp.elseBlock()) {
      idsVec.push_back(elseIds);
      blockVec.push_back(ifOp.elseBlock());
  }


  //if (ifOp.elseBlock())
  //    ifOp->emitError("Outermost is_warpgroup should not have else block\n");

  for (int i = 0; i < idsVec.size(); i++){

    DenseMap<int, scf::IfOp> tasksToIfOp;

    // Stage 1
    // Clone all operations into the corresponding if blocks. If the operation
    // has multiple taskIds, it will be cloned for multiple if blocks.
    // If the original code has an IfOp, we should only clone its
    // body with the right asyncTaskId, instead of cloning the IfOp.
    for (int asyncTaskId : idsVec[i]) { // all asyncTaskId inside
      builder.setInsertionPoint(ifOp);
      // Create IfOp for each asyncTaskId.
      Value cond = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, curAsyncTaskId,
          builder.create<arith::ConstantIntOp>(loc, asyncTaskId, 32));

      auto newIfOp = builder.create<scf::IfOp>(loc, cond);
      setOpAttrWgId(newIfOp, asyncTaskId);
      tasksToIfOp[asyncTaskId] = newIfOp;

      OpBuilder taskBuilder(context);

      // Set insertion point before yieldOp.
      auto yieldOp = newIfOp.thenYield();
      taskBuilder.setInsertionPoint(yieldOp);

      // copy over the ops inside the region to the new ifop
      IRMapping mapping;
      for (Operation &op : blockVec[i]->getOperations()) { // TODO: assume no else
        SpecializeOp(&op, mapping, taskBuilder, asyncTaskId, idsVec[i]); // TODO: add parent task ids for else block
      }
      yieldOp->erase();

      LLVM_DEBUG({
        LDBG("\n\nAfter replication:");
        newIfOp->dump();
      });
    }

    // Stage 2
    // Decide if this taskId is a producer or a consumer, and create either
    // RegAllocOp or RegDeallocOp accordingly.
    for (auto ifOps : tasksToIfOp) {
      int asyncTaskId = ifOps.first;
      auto newIfOp = ifOps.second;

      OpBuilder taskBuilder(newIfOp.getContext());
      auto regAlloc = scanRegUsage(newIfOp.thenBlock(), asyncTaskId, 0, 0, numWarpgroups, hasTma); //TODO: user control

      Block &firstBlock = newIfOp.getThenRegion().front();
      Operation& firstOp = firstBlock.front();
      if (isa<RegAllocOp, RegDeallocOp>(firstOp)){
          continue;
      }
      taskBuilder.setInsertionPointToStart(&firstBlock);
      if (regAlloc.second)
        taskBuilder.create<ttxg::RegAllocOp>(
            loc, taskBuilder.getI32IntegerAttr(regAlloc.first));
      else
        taskBuilder.create<ttxg::RegDeallocOp>(
            loc, taskBuilder.getI32IntegerAttr(regAlloc.first));

    }
  }

  // Stage 3: clean up original region
  // Remove original operations that have been cloned in reverse order.
  //for (Operation& op : llvm::reverse(ifOp.thenBlock()->getOperations())){
  //    // Erase the current operation
  //    if (!op.use_empty())
  //      LLVM_DEBUG({
  //        LDBG("op has use ");
  //        op.dump();
  //      });
  //    op.erase();
  //}
  ifOp->erase();
  return;
}

#if 0
// Lower to use GetCanonicalWarpIdOp.
// In Hopper, each task is a warpgroup consisting of 4 warps.
static const int WARPS_PER_TASK = 4;
static const int THREADS_PER_TASK = 128;
void lowerGetAsyncTaskIdOp(Operation *parentOp, int numConsumerGroups) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::GetAsyncTaskIdOp op) {
    auto loc = op.getLoc();
    OpBuilder builder(op);
    Value _4 = builder.create<arith::ConstantIntOp>(loc, WARPS_PER_TASK, 32);
    Value warpId = builder.create<ttng::GetCanonicalWarpIdOp>(loc);
    Value asyncTaskId = builder.create<arith::DivUIOp>(loc, warpId, _4);
    op.getResult().replaceAllUsesWith(asyncTaskId);

    LLVM_DEBUG({
      LDBG("erasing GetAsyncTask");
      op->dump();
    });
    eraseOps.insert(op);
  });
  for (Operation *op : eraseOps)
    op->erase();
}
#endif


class TXLGPUWSCodePartitionPass
    : public impl::TXLGPUWSCodePartitionBase<
          TXLGPUWSCodePartitionPass> {
public:
  using impl::TXLGPUWSCodePartitionBase<TXLGPUWSCodePartitionPass>::TXLGPUWSCodePartitionBase;

  // Cleanup convert ops.
  void codePartition() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // TODO: too naive
    bool hasTma = false;
    m->walk([&](tt::TmaLoadOp tmaLoadOp) {
        hasTma = true;
    });

    m->walk([&](tt::FuncOp funcOp) {
        // Find the outermost if with warp group condition
        SmallVector<scf::IfOp> wgOps = findOuterWarpGroupIf(funcOp);
        for (scf::IfOp outerIf : wgOps) {
            LLVM_DEBUG({
              DBGS() << "found wg outer if:\n";
              outerIf->dump();
            });
        }
        for (scf::IfOp outerIf : wgOps) {
          SpecializeOuterWarpgroupIf(outerIf, numWarpgroups, hasTma);
        }

    });

    m->walk([&](tt::IsWarpgroupOp isWarpgroupOp) {
        assert(isWarpgroupOp->use_empty() && "is_warpgroup not lowered!\n");
        isWarpgroupOp->erase();
    });
    OpBuilder builder(m);
    m->setAttr("ttg.total-num-warps", builder.getI32IntegerAttr(numWarpgroups*4));
    if (numWarpgroups == 1)
        m->setAttr("ttg.txl-warpgroups-set", builder.getI32IntegerAttr(0));
    else
        m->setAttr("ttg.txl-warpgroups-set", builder.getI32IntegerAttr(1));

    //lowerGetAsyncTaskIdOp(m, numWarpgroups-1); // assume only 1 producer

    LLVM_DEBUG({
      DBGS() << "Module after ws code parition:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    codePartition();
  }

};

} // namespace mlir::triton::txlgpu
