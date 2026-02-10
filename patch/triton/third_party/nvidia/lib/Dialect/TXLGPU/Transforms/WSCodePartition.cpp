#include <cstdint>
#include <cwctype>
#include <future>
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
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <deque>
#include <memory>
#include <vector>

using namespace mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttxg = mlir::triton::txlgpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttp = ::mlir::triton::proton;
namespace ttpg = ::mlir::triton::proton::gpu;

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

/*
 * IsWarpgroupOp
 */

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

/*
 * DONE IsWarpgroupOp
 */

/*
 * IsWarpOp
 */

bool hasIsWarpOperand(Value value) {
  // Base case: this value is defined by a warpgroup op
  if (auto op = value.getDefiningOp()) {
    if (isa<tt::IsWarpOp>(op)) {
      return true;
    }
  }

  // For operation results, check all operands of the defining op
  if (auto op = value.getDefiningOp()) {
    for (Value operand : op->getOperands()) {
      if (hasIsWarpOperand(operand)) {
        return true;
      }
    }
  }

  return false;
}

bool isWarpIf(Operation* op) {
    if (isa<scf::IfOp>(op)){
        scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
        Value cond = ifOp.getCondition();
        if (auto defOp = cond.getDefiningOp()) {
          if (isa<tt::IsWarpOp>(defOp)) {
            return true;
          }
          if (hasIsWarpOperand(cond))
              op->emitError("is_warp should always be used directly in if condition!\n");
          //assert(!hasWarpgroupOperand(cond) && "is_warpgroup should always be used directly in if condition!\n");
        }
    }
    return false;
}

SmallVector<scf::IfOp> findOuterWarpIf(Operation* op) {
  SmallVector<scf::IfOp> wpOps;

  // Recursively check nested operations
  for (Region& region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation& op : block) {
        if (isWarpIf(&op)) {
          wpOps.push_back(dyn_cast<scf::IfOp>(op));
        }
      }
    }
  }
  return wpOps;
}

/*
 * DONE IsWarpOp
 */


Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilder &builder,
                        SmallVector<int32_t> asyncTaskIds, llvm::ArrayRef<int32_t> rootTaskIds, SpecMode mode);

// Currently only for is_warpgroup
Operation *SpecializeSpecializedIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilder &builder,
                          int asyncTaskId, llvm::ArrayRef<int32_t> rootTaskIds, SpecMode mode) {
  LLVM_DEBUG({
    LDBG("specialize specialized ifOp ");
  });
  Value cond = ifOp.getCondition();
  ArrayRef<int32_t> ids;
  if (mode == SpecMode::WARPGROUP) {
      assert(isa<tt::IsWarpgroupOp>(cond.getDefiningOp()) && "not a isWarpgroup If\n");
      auto warpgroupOp = dyn_cast<tt::IsWarpgroupOp>(cond.getDefiningOp());
      ids = warpgroupOp.getIds();
  }
  else {
      assert(isa<tt::IsWarpOp>(cond.getDefiningOp()) && "not a isWarp If\n");
      auto warpOp = dyn_cast<tt::IsWarpOp>(cond.getDefiningOp());
      ids = warpOp.getIds();
  }


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
    SpecializeOp(&selectedOp, mapping, builder, {asyncTaskId}, selectedTaskIds, mode);
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
                          SmallVector<int32_t> asyncTaskIds, llvm::ArrayRef<int32_t> rootTaskIds, SpecMode mode) {
  LLVM_DEBUG({
    LDBG("specialize ifOp ");
    ifOp.dump();
  });

  auto loc = ifOp->getLoc();

  // Step1: result type copy to newIf
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

  // Step2: process for wids new cond
  Value cond = mapping.lookup(ifOp.getCondition());
  SmallVector<int32_t> ids;
  bool useIsWarp = isa<tt::IsWarpOp>(cond.getDefiningOp());
  if (useIsWarp) {
      assert(mode==SpecMode::WARP && "IsWarpOp mus be used solely without IsWarpgroupOp");
      auto warpOp = dyn_cast<tt::IsWarpOp>(cond.getDefiningOp());
      auto wids = warpOp.getIds();
      ids = SmallVector<int32_t>{wids.begin(), wids.end()};
      auto curAsyncTaskId = builder.create<ttxg::WarpIdOp>(loc, builder.getI32Type());
      Value geLower = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, curAsyncTaskId,
          builder.create<arith::ConstantIntOp>(loc, ids[0], 32));
      Value leUpper = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, curAsyncTaskId,
          builder.create<arith::ConstantIntOp>(loc, ids[ids.size()-1], 32));
      cond = builder.create<arith::AndIOp>(loc, geLower, leUpper);
  }

  // Step3 set op attr of wgid/wids
  // Mind the builder insertion point
  // TODO: elseblock direct copy?
  SmallVector<int32_t> thenVec(asyncTaskIds.begin(), asyncTaskIds.end());
  SmallVector<int32_t> elseVec(asyncTaskIds.begin(), asyncTaskIds.end());
  auto newIfOp = builder.create<scf::IfOp>(
      loc, newResultTypes, cond, true,
      ifOp.elseBlock());
  if (mode == SpecMode::WARPGROUP) {
      assert(asyncTaskIds.size() == 1 && "asyncTaskIds for Warpgroup should always has 1 element");
      setOpAttrWgId(newIfOp, asyncTaskIds[0]);
  }
  else if(mode == SpecMode::WARP && useIsWarp){
      // update the wids and else wids
      thenVec=ids;
      std::vector<int32_t> thenIds(thenVec.begin(), thenVec.end());
      elseVec.clear();
      subtractArrayRefs(asyncTaskIds, thenIds, elseVec);
      std::vector<int32_t> elseIds(elseVec.begin(), elseVec.end());
      setOpAttrWIds(newIfOp, thenIds);
      setOpAttrWIds(newIfOp, elseIds, true);
  }

  OpBuilder ifBuilder(ifOp.getContext());

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, thenVec, rootTaskIds, mode);
  }
  // Similarly: Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, elseVec, rootTaskIds, mode);
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
                           SmallVector<int32_t> asyncTaskIds, llvm::ArrayRef<int32_t> rootTaskIds, SpecMode mode) {

  LLVM_DEBUG({
    LDBG("specialize forOp ");
  });
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
  if (mode == SpecMode::WARPGROUP) {
    assert(asyncTaskIds.size() == 1 && "asyncTaskIds for Warpgroup should always has 1 element");
    setOpAttrWgId(newForOp, asyncTaskIds[0]);
  }
  else {
    std::vector<int32_t> vec(asyncTaskIds.begin(), asyncTaskIds.end());
    setOpAttrWIds(newForOp, vec);
  }
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
    SpecializeOp(&op, mapping, forBuilder, asyncTaskIds, rootTaskIds, mode);
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
    if (mode == SpecMode::WARPGROUP) {
      setOpAttrWgId(newYieldOp, asyncTaskIds[0]);
    }
    else {
      std::vector<int32_t> vec(asyncTaskIds.begin(), asyncTaskIds.end());
      setOpAttrWIds(newYieldOp, vec);
    }
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
                        SmallVector<int32_t> asyncTaskIds, llvm::ArrayRef<int32_t> rootTaskIds, SpecMode mode) {
  LLVM_DEBUG({
    LDBG("specialize General Ops ");
  });
  if (op->getNumRegions() == 0) {
    // case 1: direct clone
    Operation *newOp = builder.clone(*op, mapping);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    if (mode == SpecMode::WARPGROUP) {
      assert(asyncTaskIds.size() == 1 && "asyncTaskIds for Warpgroup should always has 1 element");
      setOpAttrWgId(newOp, asyncTaskIds[0]);
    }
    else {
      std::vector<int32_t> vec(asyncTaskIds.begin(), asyncTaskIds.end());
      setOpAttrWIds(newOp, vec);
    }
    return newOp;
  } else {
    // case 2: ops with blocks
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (isWarpgroupIf(ifOp)) { // only for warpgroup if
          return SpecializeSpecializedIfOp(ifOp, mapping, builder, asyncTaskIds[0], rootTaskIds, mode);
      } else {
          return SpecializeIfOp(ifOp, mapping, builder, asyncTaskIds, rootTaskIds, mode);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskIds, rootTaskIds, mode);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(newOp, asyncTaskIds[0]);
      }
      else {
        std::vector<int32_t> vec(asyncTaskIds.begin(), asyncTaskIds.end());
        setOpAttrWIds(newOp, vec);
      }
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

template <typename T, typename OP> bool hasOperator(T *o) {
  bool exist = false;
  o->walk([&](OP op) {
    exist = true;
    return WalkResult::interrupt();
  });
  return exist;
}


void SpecializeOuterSpecializedIf(scf::IfOp ifOp, int32_t numAsyncIds, bool hasTma, SpecMode mode) {

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG("Start specializing outer if region");
  });

  MLIRContext *context = ifOp.getContext();
  OpBuilder builder(ifOp);
  auto loc = ifOp.getLoc();
  Value cond = ifOp.getCondition();

  Value curAsyncTaskId;
  ArrayRef<int32_t> ids;
  if (mode == SpecMode::WARPGROUP) {
      curAsyncTaskId = builder.create<ttxg::CanonicalWarpgroupIdOp>(loc, builder.getI32Type());
      //Value curAsyncTaskId = builder.create<ttng::GetAsyncTaskIdOp>(loc);
      assert(isa<tt::IsWarpgroupOp>(cond.getDefiningOp()) && "not a isWarpgroup If\n");
      auto warpgroupOp = dyn_cast<tt::IsWarpgroupOp>(cond.getDefiningOp());
      ids = warpgroupOp.getIds();
  }
  else {
      curAsyncTaskId = builder.create<ttxg::WarpIdOp>(loc, builder.getI32Type());
      assert(isa<tt::IsWarpOp>(cond.getDefiningOp()) && "not a isWarp If\n");
      auto warpOp = dyn_cast<tt::IsWarpOp>(cond.getDefiningOp());
      ids = warpOp.getIds();
  }


  SmallVector<int32_t> rootTaskIds;
  numAsyncIds = nextPowerOf2(numAsyncIds);
  for (int32_t i=0; i < numAsyncIds; i++)
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

  if (mode == SpecMode::WARPGROUP) {
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
          tasksToIfOp[asyncTaskId] = newIfOp;

          setOpAttrWgId(newIfOp, asyncTaskId);
          newIfOp->setAttr("txl-outer-warpgroup", builder.getI32IntegerAttr(1));

          OpBuilder taskBuilder(context);

          // Set insertion point before yieldOp.
          auto yieldOp = newIfOp.thenYield();
          taskBuilder.setInsertionPoint(yieldOp);

          // copy over the ops inside the region to the new ifop
          IRMapping mapping;
          for (Operation &op : blockVec[i]->getOperations()) {
            SpecializeOp(&op, mapping, taskBuilder, {asyncTaskId}, idsVec[i], mode);
          }

          // TODO: to enable yield we either:
          // 1. yield dummy for else
          // 2. nest if else in else
          //unsigned resultIdx = 0;
          //SmallVector<unsigned> keptResultVec;
          //if (!ifOp->getResultTypes().empty()) {
          //  for (Value yieldV : ifOp.thenYield()->getOperands()) {
          //    keptResultVec.push_back(resultIdx);
          //    ++resultIdx;
          //  }
          //}

          //SmallVector<Type> newResultTypes;
          //for (auto idx : keptResultVec) {
          //  newResultTypes.push_back(ifOp->getResultTypes()[idx]);
          //}

          //for (auto idx : keptResultVec) {
          //  auto oldIfResultSrc = ifOp.thenYield()->getOperands()[idx]; // then or else
          //  auto flatIfResult = mapping.lookupOrDefault(oldIfResultSrc);
          //  assert(flatIfResult && "Unexpected missing mapping");

          //  mapping.map(ifOp.getResult(idx), flatIfResult);
          //}

          yieldOp->erase();

          LLVM_DEBUG({
            LDBG("\n\nAfter replication:");
            newIfOp->dump();
          });
        }


        // Stage 2
        // Decide if this taskId is a producer or a consumer, and create either
        // ONLY For Warpgroups
        // RegAllocOp or RegDeallocOp accordingly.
        for (auto ifOps : tasksToIfOp) {
          int asyncTaskId = ifOps.first;
          auto newIfOp = ifOps.second;

          OpBuilder taskBuilder(newIfOp.getContext());
          auto regAlloc = scanRegUsage(newIfOp.thenBlock(), asyncTaskId, 0, 0, numAsyncIds, hasTma); //TODO: user control

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

        // Stage 3
        // check proton record op exist and add ctx ops
        for (auto ifOps : tasksToIfOp) {
          int asyncTaskId = ifOps.first;
          // don't create new segment for master warp
          if (asyncTaskId == 0)
              continue;
          auto newIfOp = ifOps.second;

          OpBuilder taskBuilder(newIfOp.getContext());

          Block &firstBlock = newIfOp.getThenRegion().front();

          if (hasOperator<Operation, ttp::RecordOp>(newIfOp)) {
              taskBuilder.setInsertionPointToStart(&firstBlock);
              taskBuilder.create<ttxg::RestoreCtxOp>(loc, asyncTaskId);

              taskBuilder.setInsertionPoint(newIfOp.thenYield());
              taskBuilder.create<ttxg::SaveCtxOp>(loc, asyncTaskId);
          }

        }
    }
  }
  else { // SpecMode.WARP
    builder.setInsertionPoint(ifOp);
    // Create IfOp for each asyncTaskId.
    Value geLower = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, curAsyncTaskId,
        builder.create<arith::ConstantIntOp>(loc, idsVec[0][0], 32));
    Value leUpper = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, curAsyncTaskId,
        builder.create<arith::ConstantIntOp>(loc, idsVec[0][idsVec[0].size()-1], 32));
    Value cond = builder.create<arith::AndIOp>(loc, geLower, leUpper);

    // TODO: assumes no result type
    auto newIfOp = builder.create<scf::IfOp>(loc, cond, idsVec.size()>1);
    //tasksToIfOp[asyncTaskId] = newIfOp;

    setOpAttrWIds(newIfOp, idsVec[0]);
    if (idsVec.size() > 1)
      setOpAttrWIds(newIfOp, idsVec[1], true);
    newIfOp->setAttr("txl-outer-warp", builder.getI32IntegerAttr(1));

    OpBuilder taskBuilder(context);

    // Set insertion point before yieldOp.
    auto yieldOp = newIfOp.thenYield();
    taskBuilder.setInsertionPoint(yieldOp);

    // copy over the ops inside the region to the new ifop
    IRMapping mapping;
    for (Operation &op : blockVec[0]->getOperations()) {
      SmallVector<int32_t> vec{idsVec[0].begin(), idsVec[0].end()};
      SpecializeOp(&op, mapping, taskBuilder, vec, idsVec[0], mode);
    }
    yieldOp->erase();

    if (blockVec.size() > 1) {
      auto elseYieldOp = newIfOp.elseYield();
      taskBuilder.setInsertionPoint(elseYieldOp);
      for (Operation &op : blockVec[1]->getOperations()) {
        SmallVector<int32_t> vec{idsVec[1].begin(), idsVec[1].end()};
        SpecializeOp(&op, mapping, taskBuilder, vec, idsVec[1], mode);
      }
      elseYieldOp.erase();

      Block* elseBlock = blockVec[1];

    }
    if (hasOperator<Operation, ttp::RecordOp>(newIfOp)) {
        for (int i = 0; i<idsVec.size(); i++) {
            for (auto asyncTaskId : idsVec[i]) {
              if (asyncTaskId != 0) {
                  taskBuilder.setInsertionPointToStart(blockVec[i]);
                  taskBuilder.create<ttxg::RestoreCtxOp>(loc, asyncTaskId);

                  if (i == 0)
                      taskBuilder.setInsertionPoint(newIfOp.thenYield());
                  else
                      taskBuilder.setInsertionPoint(newIfOp.elseYield());
                  taskBuilder.create<ttxg::SaveCtxOp>(loc, asyncTaskId);
              }
            }
        }
    }

    LLVM_DEBUG({
      LDBG("\n\nAfter replication:");
      newIfOp->dump();
    });
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

    llvm::DenseSet<int32_t> allWarps;
    m->walk([&](tt::IsWarpOp isWarpOp) {
        auto ids = isWarpOp.getIds();
        for (auto i : ids) {
          allWarps.insert(i);
        }
    });
    int32_t numAllWarps = allWarps.size();

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
          SpecializeOuterSpecializedIf(outerIf, numWarpgroups, hasTma, SpecMode::WARPGROUP); // hasTma is used for heuristic reg alloc
        }
        // Find the outermost if with warp condition
        SmallVector<scf::IfOp> wOps = findOuterWarpIf(funcOp);
        for (scf::IfOp outerIf : wOps) {
            LLVM_DEBUG({
              DBGS() << "found warp outer if:\n";
              outerIf->dump();
            });
        }
        for (scf::IfOp outerIf : wOps) {
          SpecializeOuterSpecializedIf(outerIf, numAllWarps, hasTma, SpecMode::WARP); // hasTma is used for heuristic reg alloc
        }

        // add init context
        if ((wgOps.size() || wOps.size()) && hasOperator<Operation, ttp::RecordOp>(funcOp)) {
            OpBuilder builder(funcOp);
            builder.setInsertionPointToStart(&funcOp.getBody().front());
            builder.create<ttxg::InitCtxOp>(funcOp.getLoc()); // TODO: better loc
        }
    });

    LLVM_DEBUG({
      DBGS() << "Module after ws code parition:\n";
      DBGS() << printModuleOp(m);
    });

    llvm::SmallDenseSet<int32_t> widsSet;
    m->walk([&](triton::IsWarpOp warpOp) {
      auto wids = warpOp.getIds();
      assert(wids.size() && "IsWarpOp expected to have ids attribute");

      for (auto id: wids) {
        widsSet.insert(id);
      }
    });
    size_t numUniqueWIds = widsSet.size();

    m->walk([&](tt::IsWarpgroupOp isWarpgroupOp) {
        assert(isWarpgroupOp->use_empty() && "is_warpgroup not lowered!\n");
        isWarpgroupOp->erase();
    });
    m->walk([&](tt::IsWarpOp isWarpOp) {
        assert(isWarpOp->use_empty() && "is_warp not lowered!\n");
        isWarpOp->erase();
    });

    OpBuilder builder(m);
    if (numUniqueWIds == 0)
        m->setAttr("ttg.total-num-warps", builder.getI32IntegerAttr(numWarpgroups*4));
    else {
        m->setAttr("ttg.total-num-warps", builder.getI32IntegerAttr(numUniqueWIds));
        llvm::outs() << "\n total num warps \n";
        llvm::outs() << numUniqueWIds;
        llvm::outs() << "\n";
    }
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
