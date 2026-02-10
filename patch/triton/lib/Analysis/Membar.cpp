#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/TXLUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <cstdint>
#include <deque>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

// txl
static inline void insertBarrierTXL(OpBuilder &builder, Operation *op) {
  //op->emitError("Insert Barrier");
  //op->dump();
  //llvm::outs() << "\n";
  auto barrierOp = builder.create<mlir::gpu::BarrierOp>(op->getLoc());
  auto attr = triton::getParentWithWGIDAttr(op);

  auto mod = op->getParentOfType<ModuleOp>();
  int numWarps = mlir::triton::gpu::lookupNumWarps(op);
  int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  int nameBarrierIdBegin = 1; // TODO align with python
  int nameBarrierIdEnd = 8; // python should start from 8
  if (attr) {
    assert(attr.getType().isInteger(32) && "ttxg.wgid must be 32 bit int\n");
    int32_t asyncTaskId = attr.getInt();

    int barId = asyncTaskId + nameBarrierIdBegin;
    assert(barId < nameBarrierIdEnd);
    // NOTE: assume numWarps is always 4 in this case.
    int numThreads = 4 * warpSize;
    barrierOp->setAttr("bar_id", builder.getI64IntegerAttr(barId));
    barrierOp->setAttr("num_threads", builder.getI64IntegerAttr(numThreads));
  }

  SmallVector<int32_t> wids = triton::getParentWithWIDsAttr(op);
  if (wids.size()) {
    int barId = wids[0] + nameBarrierIdBegin; // NOTE: assume sorted wids
    assert(barId < nameBarrierIdEnd);
    // NOTE: assume numWarps is always 4 in this case.
    int numThreads = wids.size() * warpSize;
    barrierOp->setAttr("bar_id", builder.getI64IntegerAttr(barId));
    barrierOp->setAttr("num_threads", builder.getI64IntegerAttr(numThreads));
  }
}

bool isTritonWrongSync(Operation* op){
    bool isWrong =  isa<ttng::InitBarrierOp>(op); // smem_index ignored offset, making overlap
    isWrong = isWrong || isa<ttng::AsyncTMACopyGlobalToLocalOp>(op); // tma_load + expect, as WAW
    isWrong = isWrong || isa<ttng::WaitBarrierOp>(op); // wait on same mbar, treated as WRAWR
    isWrong = isWrong || isa<ttng::BarrierExpectOp>(op); // init->expect, expect->expect, being WAW
    // isWrong = isWrong || isa<ttng::AsyncTMACopyLocalToGlobalOp>(op); // tma_store with local_alloc necessary
    return isWrong;
}

Interval<size_t> getWGBufferInterval(Interval<size_t> range, Operation* op){
  auto attr = triton::getParentWithWGIDAttr(op);

  if (attr) {
    assert(attr.getType().isInteger(32) && "ttxg.wgid must be 32 bit int\n");
    int32_t asyncTaskId = attr.getInt();
    auto newOff = asyncTaskId * 300000;// TODO: hardcoded
    Interval<size_t> newRange(range.start() + newOff, range.end() + newOff);
    return newRange;
  }
  SmallVector<int32_t> wids = triton::getParentWithWIDsAttr(op);
  if (wids.size()) {
    int32_t asyncTaskId = wids[0];
    auto newOff = asyncTaskId * 300000;// TODO: hardcoded
    Interval<size_t> newRange(range.start() + newOff, range.end() + newOff);
    return newRange;
  }
  return range;
}

void MembarOrFenceAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarOrFenceAnalysis::resolve(FunctionOpInterface funcOp,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  // Initialize the blockList. Operations are organized into "virtual blocks",
  // which represent segments of straight-line code analyzed by each iteration
  // of the dataflow analysis. Virtual blocks abstract over both control flow
  // represented by basic blocks and block successors (i.e. `BranchOpInterface`)
  // and control flow represented by regions (i.e. `RegionBranchOpInterface`).
  //
  // A virtual block consists of a parent block and a starting iterator, where
  // the virtual block starts on the operation *after* the starting iterator. A
  // null iterator is used to represent the beginning of the block. The virtual
  // block ends at any region branch operation or the basic block terminator.
  // Thus, basic blocks are broken up into multiple virtual blocks at each
  // region operation.
  //
  // Entry virtual blocks are represented by a null iterator. Populate the
  // blockList with the entry virtual blocks in the function. Then, each
  // iteration scans until a terminator or region branch operation is found.
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    // Start the analysis from the entry blocks of any nested isolated from
    // above regions.
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      blockList.emplace_back(block, Block::iterator());
  });

  // A fixed point algorithm
  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        block.second.isValid() ? std::next(block.second) : block.first->begin();
    for (Operation &op : llvm::make_range(startIt, block.first->end())) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block. The block transfer function is not monotonic,
    // so overwrite the output state entirely.
    outputBlockInfoMap[block] = inputBlockInfo;
    // Update the successors
    for (VirtualBlock successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](triton::ReturnOp returnOp) {
    // A basic block can be broken into several virtual blocks. Find all virtual
    // blocks that belong to the basic block containing the return.
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[block, blockInfo] : outputBlockInfoMap) {
      if (block.first == returnOp->getBlock())
        virtualBlocks.emplace_back(block, blockInfo);
    }
    // The return is a terminator, so the virtual block that contains this
    // return starts after all other ones. Find it by comparing the start
    // iterators of the virtual blocks.
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      assert(lhs.first.first == rhs.first.first);
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });

    funcBlockInfo.join(maxIt->second);
  });
}

void MembarOrFenceAnalysis::visitTerminator(
    Operation *op, SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // The successors of an operation with regions can be queried via an
    // interface. The operation branches to the entry blocks of its region
    // successors. It can also branch to after itself.
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        successors.emplace_back(br->getBlock(), br->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
  // reason. Check that the parent is actually a `RegionBranchOpInterface`.
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    // Check the successors of a region branch terminator. It can branch to
    // another region of its parent operation or to after the parent op.
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // Otherwise, it could be a return op
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

// txl
void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  insertBarrierTXL(*builder, op);
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  // txl: what if is return, br?
  //if (isa<triton::gpu::AsyncWaitOp, triton::nvidia_gpu::TMAStoreWaitOp>(op) &&
  //    !isa<gpu::BarrierOp>(op->getNextNode())) {
  //  // If the current op is an async wait and the next op is not a barrier we
  //  // insert a barrier op and sync
  //  builder->setInsertionPointAfter(op);
  //  insertBarrier(op, builder);
  //  blockInfo->sync();
  //  return;
  //}

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!triton::isFakeMemoryEffects(op)){

      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo
                    .syncWriteIntervals[getWGBufferInterval(allocation->getAllocatedInterval(
                        bufferId), op)]
                    .insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo
                    .syncReadIntervals[getWGBufferInterval(allocation->getAllocatedInterval(
                        bufferId), op)]
                    .insert(op);
            }
          }
        }
      }

      }
    }
    // If this op is may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
    //if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
    //  Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
    //  curBlockInfo.syncWriteIntervals[allIntervals].insert(op);
    //  curBlockInfo.syncReadIntervals[allIntervals].insert(op);
    //}
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    // Detect warp-synchronous convert-layout operations. These emit a
    // warp-level barrier (warp.sync) rather than a CTA-wide barrier between
    // the internal shared-memory write and read phases. For these ops, we must
    // not globally clear pending dependencies.
    bool isWarpSync = false;
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(cvt.getType());
      auto srcLayout = triton::gpu::toLinearLayout(srcTy);
      auto dstLayout = triton::gpu::toLinearLayout(dstTy);
      isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    }

    if (!curBlockInfo.syncReadIntervals.empty() ||
        !curBlockInfo.syncWriteIntervals.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = getWGBufferInterval(allocation->getAllocatedInterval(scratchBufferId), op);
    curBlockInfo.syncWriteIntervals[interval].insert(op);
    auto insertCTABarrier = blockInfo->isIntersected(curBlockInfo, filter);
    if (insertCTABarrier && !isTritonWrongSync(op)) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    // Ops with a scratch buffer that don't use warp.sync internally sync
    // read/write on shared memory
    if (insertCTABarrier || !isWarpSync)
      blockInfo->sync();
    curBlockInfo.syncReadIntervals[interval].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter) && !isTritonWrongSync(op)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
