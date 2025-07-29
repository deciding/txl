#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "txlgpu-wgmma-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value assigned to the argument coming from outside the loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  Value val = waitOp->getOperand(0);
  // If the value resides in a region other than the region of the wait op, then
  // the wait op must be in some nested region. Measure the number of commits
  // between the definition value and the parent op.
  // TODO: We could measure commits in nested regions along the path if
  // necessary.
  while (waitOp->getParentRegion() != val.getParentRegion())
    waitOp = waitOp->getParentOp();
  int minCommits = minOverHistories(val, waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && !isa<ttg::AsyncCommitGroupOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.front());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::txlgpu::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<ttg::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

static void lowerDotWaitInner(SmallVector<Operation*> &queue, SmallVector<std::pair<Value, int>> &asyncDotResults) {
    while(!queue.empty()){
        Operation * curOp = queue.pop_back_val();
        if (isa<scf::IfOp>(curOp)){
            scf::IfOp ifOp = dyn_cast<scf::IfOp>(curOp);

            SmallVector<Operation*> thenQueue;
            for (Operation& op : llvm::reverse(ifOp.thenBlock()->getOperations())){
                thenQueue.push_back(&op);
            }
            SmallVector<std::pair<Value, int>> thenAsyncDotResults(asyncDotResults.begin(), asyncDotResults.end());
            lowerDotWaitInner(thenQueue, thenAsyncDotResults);

            if (ifOp.elseBlock()){
                SmallVector<Operation*> elseQueue;
                for (Operation& op : llvm::reverse(ifOp.elseBlock()->getOperations())){
                    elseQueue.push_back(&op);
                }
                SmallVector<std::pair<Value, int>> elseAsyncDotResults(asyncDotResults.begin(), asyncDotResults.end());
                lowerDotWaitInner(elseQueue, elseAsyncDotResults);
            }
        }
        else if (isa<ttng::WarpGroupDotOp>(curOp)){
            ttng::WarpGroupDotOp dotOp = dyn_cast<ttng::WarpGroupDotOp>(curOp);
            int opNum = -1;
            for (auto& use : dotOp->getUses()){
                auto user = use.getOwner();
                if (isa<scf::YieldOp>(user)){
                    auto yieldOp = dyn_cast<scf::YieldOp>(user);
                    opNum = use.getOperandNumber();
                }
            }

            asyncDotResults.push_back({curOp->getResult(0), opNum});
        }
        else if (isa<tt::WGDotWaitOp>(curOp)){
            OpBuilder builder(curOp);
            tt::WGDotWaitOp waitOp = dyn_cast<tt::WGDotWaitOp>(curOp);
            uint32_t pendings = waitOp.getPendings();

            builder.setInsertionPoint(waitOp);
            auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
                waitOp->getLoc(), ArrayRef<Value>{}, pendings);

            auto waitStart = asyncDotResults.begin();
            int numWaiting = asyncDotResults.size() - pendings;
            auto waitEnd = waitStart + numWaiting;

            SmallVector<Value> waitDotResults; // originally all will be thread through

            auto tmpWaitEnd = waitEnd;
            if (waitStart == waitEnd)
                tmpWaitEnd ++; // TODO: tmp logic, dot_wat must have operands?

            for (auto it = waitStart; it != tmpWaitEnd; ++it){
                auto [val, opNum] = *it;
                waitDotResults.push_back(val);
            }

            threadValuesThroughWait(wait, waitDotResults);

            asyncDotResults.erase(waitStart, waitEnd);

            waitOp->erase();
        }
    }
}

// TXL: 
// stage1: from all ops in forOp: 1. direct child of forOp, 2. forget about ifOp, do aggressively
//        encouter asyncDot, add to pendingAccumDotBuffer
//        encouter WGDotWaitOp 0, clear pendingAccumDotBuffer
//        if WGDotWaitOp N, clear before last $pendings
//        at last check whether they have yieldOp in forOp, and record the iterArg
// stage2: from all ops in forOp: 1. direct child of forOp, 2. any nested uses in ifOp
//        first prepend the pendingAccum iterArg
//        encouter asyncDot, add to asyncDotBuffer
//        encouter WGDotWaitOp 0, add all results from asyncDotBuffer, clear asyncDotBuffer
//        if WGDotWaitOp N, pick $pendings from asyncDotBuffer (max size), clear them
//        correctly enqueue and pop for then/else, also enqueu new asyncDotBuffer
// stage3: for the end outof forOp, add wait<0> on remaining asyncDotBuffer
//
static SmallVector<std::pair<Value, int>> lowerDotWait(scf::ForOp forOp) {
  SmallVector<Operation*> pendingAccumDotBuffer;

  // Stage 1
  for (auto& op : forOp.getBody()->getOperations()){
      if (isa<ttng::WarpGroupDotOp>(op)){
          pendingAccumDotBuffer.push_back(&op);
      }
      if (isa<tt::WGDotWaitOp>(op)){
          auto wgDotWaitOp = dyn_cast<tt::WGDotWaitOp>(op);
          uint32_t pendings = wgDotWaitOp.getPendings();
          auto erase_start = pendingAccumDotBuffer.begin();
          if (pendings >= pendingAccumDotBuffer.size())
              continue;
          auto erase_end = pendingAccumDotBuffer.begin() + (pendingAccumDotBuffer.size() - pendings);
          pendingAccumDotBuffer.erase(erase_start, erase_end);
      }

  }

  SmallVector<std::pair<Value, int>> pendingArgBuffer;
  for (auto op : pendingAccumDotBuffer){
      for (auto& use : op->getUses()){
          auto user = use.getOwner();
          if (isa<scf::YieldOp>(user)){
              auto yieldOp = dyn_cast<scf::YieldOp>(user);
              int opNum = use.getOperandNumber();
              Value iterArg = forOp.getRegionIterArg(opNum);
              pendingArgBuffer.push_back({iterArg, opNum});
          }
      }
  }

  Block* forBlock = forOp.getBody();
  SmallVector<Operation*> queue;
  for (auto it = forBlock->rbegin(), end = forBlock->rend(); it!=end; it++) {
      queue.push_back(&*it);
  }
  SmallVector<std::pair<Value, int>> asyncDotResults = pendingArgBuffer;

  // Stage 2
  lowerDotWaitInner(queue, asyncDotResults);

  return asyncDotResults;
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
//
// TXL: this is only for the loop direct child ops, the recursion is done on calling side
// it also consider the nested ifOps as deep as you want.
void triton::txlgpu::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());

  SmallVector<std::pair<Value, int>> remainingDotResults = lowerDotWait(forOp);

  LLVM_DEBUG({
    LDBG("lowerDotWait\n");
    forOp->dump();
    LDBG("DONE\n\n\n");
  });

  if (remainingDotResults.size() == 0)
      return;


  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  DenseSet<int> seenOperands;

  for (auto [res, iterArgIdx] : remainingDotResults){
      if (seenOperands.insert(iterArgIdx).second && iterArgIdx != -1){
          waitOperands.push_back(forOp.getResult(iterArgIdx));
      }
  }
  // TXL: the final wait<0> after forOp is needed
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto WarpGroupDotWaitAfterLoop = builder.create<ttng::WarpGroupDotWaitOp>(
      forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);

  LLVM_DEBUG({
    LDBG("forLoop end wait\n");
    forOp->dump();
    LDBG("DONE\n\n\n");
  });

}
