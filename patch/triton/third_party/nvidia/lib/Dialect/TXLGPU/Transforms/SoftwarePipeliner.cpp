#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace txlgpu {

#define GEN_PASS_DEF_TXLGPUPIPELINE
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-pipeliner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static void pipelineWgmma(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    mlir::triton::txlgpu::asyncLaunchDots(forOp);
  }
  moduleOp->walk([&](tt::WGDotWaitOp waitOp) { waitOp.erase(); });
}

bool isTMALoad(Operation *op) {
  return isa<tt::TmaLoadOp>(op);
}

// smem_alloc's tma_load users
// TODO: assume all the same, if we want to reuse smem, just dealloc all and realloc
template<typename OpTy>
SmallVector<OpTy> getSmemAllocUsers(Operation *op) {
  SmallVector<OpTy> ops;
  for (auto user0 : op->getUsers()) {
    if (isa<OpTy>(user0)) {
      ops.push_back(dyn_cast<OpTy>(user0));
    }

    if (isa<tt::GetBufferOp>(user0)) {
      for (auto user : user0->getUsers()) {
        if (isa<OpTy>(user)) {
          ops.push_back(dyn_cast<OpTy>(user));
        }
      }
    }
  }
  return ops;
}

template<typename OpTy>
OpTy getSmemAllocRoot(Value &smem){
    auto op = smem.getDefiningOp<OpTy>();
    if (op)
        return op;
    auto opBuf = smem.getDefiningOp<tt::GetBufferOp>();
    if (opBuf){
        auto src = opBuf.getSrc();
        auto rootOp = src.getDefiningOp<OpTy>();
        if (rootOp)
            return rootOp;
    }
    return nullptr;
}

// originally op is load ops, now is smem_alloc op
ttg::SharedEncodingTrait getSharedEncoding(Operation *op) {

  // Try to use local alloc encoding if possible.
  // local_alloc(smem_alloc)
  ttg::SharedEncodingTrait localAllocEnc;
  SmallVector<ttg::LocalAllocOp> localAllocs = getSmemAllocUsers<ttg::LocalAllocOp>(op);

  if (localAllocs.size()) {
    for (ttg::LocalAllocOp localAlloc : localAllocs) {
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc; // get local_alloc result smem enc
      }
      if (enc != localAllocEnc) {
        // Some users have different encoding than others.
        // Use one of the encodings, and warn about the performance issue.
        op->emitRemark()
            << "Pipelining load with different use encodings. This will lead "
               "to layout conversions and performance degradation.";
        continue;
      }
    }
  }

  // smem_alloc result type
  auto opty = op->getResultTypes()[0];
  auto ty = cast<RankedTensorType>(opty);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto order = ttg::getOrder(ty);

  SmallVector<tt::TmaLoadOp> tmaLoads = getSmemAllocUsers<tt::TmaLoadOp>(op);

  // tma_load overwrites local_alloc enc
  if (tmaLoads.size()) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    Operation* userOp = tmaLoads[0];
    if (localAllocEnc) {
      auto sharedMMALayout =
              dyn_cast<ttg::NVMMASharedEncodingAttr>(localAllocEnc);
      if (sharedMMALayout) {
        assert(!sharedMMALayout.getFp4Padded() &&
               "TMA load for mixed precision MMAv5 is not supported yet.");
      }
    }
    auto res = ttg::NVMMASharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType(),
        /*fp4Padded*/ false);
    return res;
  }

  if (localAllocEnc)
    return localAllocEnc;

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
          .value_or(nullptr);

  if (localAllocEnc)
    return localAllocEnc;

  // Use generic layout. This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

// Make this general Operation *, thus numStages is passed in not inferred inside
static Value createAlloc(Operation* loadOp, int numStages,
                         ttg::SharedEncodingTrait sharedEnc
                         ) {
  OpBuilder builder(loadOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(loadOp->getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());

  // TODO: any condition?
  bufferShape.insert(bufferShape.begin(), numStages);

  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);

  //builder.setInsertionPointAfter(forOp);
  //builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return alloc;
}

// Create an allocation and init the mbarriers.
Value createBarrierAlloc(tt::MbarAllocOp& mbarAllocOp) {
  int arrCount = mbarAllocOp.getArrCount();
  int numBarriers = mbarAllocOp.getNumStages();
  MLIRContext *ctx = mbarAllocOp->getContext();
  IRRewriter rewriter(ctx); // OpBuilder is also okay I suppose
  rewriter.setInsertionPoint(mbarAllocOp);
  Location loc = mbarAllocOp->getLoc();
  unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
      mbarAllocOp->getParentOfType<ModuleOp>());
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = ttg::CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{numCTAs},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  ttg::MemDescType barrierMemDescType = ttg::MemDescType::get(
      {numBarriers}, rewriter.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Value barrierAlloc =
      rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = triton::createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InitBarrierOp>(loc, barrierView, arrCount);
  }
  return barrierAlloc;
}

// loadOp is tmaLoadOp, alloc is the LocalAllocOp from smem_alloc
Value lowerGetBufferOp(tt::GetBufferOp getBufferOp, Value newAlloc) {
  OpBuilder builder(getBufferOp);
  Location loc = getBufferOp->getLoc();

  Value buffer = triton::createSingleBufferView(builder, newAlloc,
                                                 getBufferOp.getIndex());
  return buffer;
}


void replaceSmemBufferUses(
    Operation* oldViewOp, Value newView, // replace oldViewOp uses with newView
    ttg::MemDescType allocTy) {

      OpBuilder builder(oldViewOp);
      Location loc = oldViewOp->getLoc();

      // remove redundant local_alloc
      SmallVector<ttg::LocalAllocOp> allocsToErase;
      for (Operation *user : oldViewOp->getUsers()) {
        if (auto userAllocOp = dyn_cast<ttg::LocalAllocOp>(user)) {
          if (allocTy.getEncoding() == userAllocOp.getType().getEncoding()) {
            // replace all uses of userAlloc to newView
            tt::replaceUsesAndPropagateType(builder, userAllocOp, newView);
            allocsToErase.push_back(userAllocOp);
          }
        }
      }
      for (auto allocOp : allocsToErase) {
        allocOp.erase();
      }

      // TmaLoadOp skipped local_load op wrappings
      // If there are some uses that were not local_allocs, we need to create a
      // local_load for them.
      for (Operation* user : oldViewOp->getUsers()){
        if (auto tmaLoadOp = dyn_cast<tt::TmaLoadOp>(user)) {
            tmaLoadOp->setOperand(0, newView);
        }
        else {
            auto sharedLoad = builder.create<ttg::LocalLoadOp>(
                loc, oldViewOp->getResultTypes().front(), newView);
            auto result = sharedLoad->getResults();
            oldViewOp->replaceAllUsesWith(result);
        }
      }

      //oldViewOp->erase();
}

// replace oldViewOp uses with newView
void replaceMbarBufferUses(Operation* oldViewOp, Value newView) {

      OpBuilder builder(oldViewOp);
      Location loc = oldViewOp->getLoc();
      tt::replaceUsesAndPropagateType(builder, oldViewOp, newView);
      //oldViewOp->erase();
}

void lowerMbarOps(Value& mbar, bool usedByTmaLoadOp) {
    auto mbarOp = mbar.getDefiningOp();
    OpBuilder builder(mbarOp);
    for (auto user : mbarOp->getUsers()){
        if (isa<tt::MbarExpectOp>(user)){
            auto mbarExpectOp = dyn_cast<tt::MbarExpectOp>(user);
            //Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
            builder.setInsertionPoint(user);
            builder.create<ttng::BarrierExpectOp>(user->getLoc(), mbar,
                    mbarExpectOp.getExpectCount(), mbarExpectOp.getPred());
            mbarExpectOp->erase();
        }
        else if (isa<tt::MbarWaitOp>(user)){
            auto mbarWaitOp = dyn_cast<tt::MbarWaitOp>(user);
            builder.setInsertionPoint(user);
            builder.create<ttng::WaitBarrierOp>(user->getLoc(), mbar,
                                                    mbarWaitOp.getPhase());
            mbarWaitOp->erase();
        }
        else if (isa<tt::MbarArriveOp>(user)){
            auto mbarArriveOp = dyn_cast<tt::MbarArriveOp>(user);
            builder.setInsertionPoint(user);
            builder.create<ttng::MBarrierArriveOp>(
                    user->getLoc(),
                    mbar,
                    mbarArriveOp.getPred(),
                    /*remoteCTAId*/ nullptr,
                    mbarArriveOp.getTrackAsyncOp(), // TODO: auto by tmaload/asyncload
                    mbarArriveOp.getTxCount()
                    );
            mbarArriveOp->erase();
        }
    }

}



void lowerSmemAlloc(tt::SmemAllocOp op){
    auto sharedEnc = getSharedEncoding(op);
    // this local_alloc is bind with this smem_alloc
    auto alloc = createAlloc(op, op.getNumStages(), sharedEnc);
    // these loads/tma_loads binds with local_alloc
    // TODO: add loads
    auto getBufferOps = getSmemAllocUsers<tt::GetBufferOp>(op);
    assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
           "Expected MemDescType");
    auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());

    Value buffer;
    for (auto getBufferOp : getBufferOps){
        buffer = lowerGetBufferOp(getBufferOp, alloc);
        replaceSmemBufferUses(getBufferOp, buffer, allocDescType);
        getBufferOp->erase();
    }
    replaceSmemBufferUses(op, alloc, allocDescType);
    op->erase();

    // for remaining MbarAllocOp
    // create new barriers
    // lower all barrier usages
    // remove all tt barrier ops
        
}

bool hasTMALoadUsers(Operation* op){
    for (auto user : op->getUsers()){
        if (isa<tt::TmaLoadOp>(user)){
            return true;
        }
    }
    return false;
}

void lowerMbar(tt::MbarAllocOp& op) {
    // For each group calculate the size and insert the barrier after the last
    // load.
    //sizeInBytes +=
    //    loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;

    OpBuilder builder(op);

    Value barrierAlloc = createBarrierAlloc(op);

    auto getBufferOps = getSmemAllocUsers<tt::GetBufferOp>(op);

    Value buffer;
    bool usedByTmaLoadOp = false;
    for (auto getBufferOp : getBufferOps){
        usedByTmaLoadOp = hasTMALoadUsers(getBufferOp);
        buffer = lowerGetBufferOp(getBufferOp, barrierAlloc);
        replaceMbarBufferUses(getBufferOp, buffer);
        getBufferOp->erase();
        lowerMbarOps(buffer, usedByTmaLoadOp); // removed
    }

    usedByTmaLoadOp = hasTMALoadUsers(op);
    replaceMbarBufferUses(op, barrierAlloc);
    op->erase();
    lowerMbarOps(barrierAlloc, usedByTmaLoadOp);
    // Invalidate and deallocate barrier
    //builder.setInsertionPointAfter(forOp);
    //for (int i = 0; i < numBuffers; i++) {
    //  Value barrierView =
    //      triton::createSingleBufferView(builder, barrierAlloc, i);
    //  builder.create<ttng::InvalBarrierOp>(loc, barrierView);
    //}
    //builder.create<ttg::LocalDeallocOp>(loc, barrierAlloc);
}

void lowerTmaLoadOp(tt::TmaLoadOp &tmaLoad) {

    Value desc = tmaLoad.getDesc();

    OpBuilder builder(tmaLoad);
    Location loc = tmaLoad->getLoc();

    auto pred = builder.create<arith::ConstantIntOp>(loc, 1, 1); // TODO: tma load may need pred
    auto tmaPtr =
        builder.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(loc, desc);

    // tmaLoad.getMbar() not working, should have an intermediate op with MemDescType
    // workaround: use getOperand instead
    builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
        tmaLoad.getLoc(), tmaPtr, tmaLoad.getIndices(),
        tmaLoad.getOperand(4), tmaLoad.getOperand(0), pred);

    tmaLoad->erase();
}

// helper
void printAllMbar(ModuleOp moduleOp){
  SmallVector<tt::GetBufferOp> getBufferOps;
  moduleOp->walk([&](tt::GetBufferOp getBufferOp) {
      getBufferOp->dump();
      llvm::outs() << "\n";
  });
}

void lowerSmemAllocs(ModuleOp moduleOp) {
  SmallVector<tt::SmemAllocOp> smemAllocs;

  moduleOp->walk([&](tt::SmemAllocOp smemAllocOp) { smemAllocs.push_back(smemAllocOp); });
  if (smemAllocs.empty())
    return;
  for (auto smemAllocOp : smemAllocs) {
    lowerSmemAlloc(smemAllocOp);
  }
}

void lowerMbars(ModuleOp moduleOp) {
  SmallVector<tt::MbarAllocOp> mbarAllocOps;
  moduleOp->walk([&](tt::MbarAllocOp mbarAllocOp) {
      mbarAllocOps.push_back(mbarAllocOp);
  });
  for (auto mbarAllocOp : mbarAllocOps) {
      lowerMbar(mbarAllocOp);
  }
}

void lowerTmaLoads(ModuleOp moduleOp) {
  SmallVector<tt::TmaLoadOp> tmaLoadOps;
  moduleOp->walk([&](tt::TmaLoadOp tmaLoadOp) {
      tmaLoadOps.push_back(tmaLoadOp);
  });
  for (auto tmaLoadOp : tmaLoadOps) {
      lowerTmaLoadOp(tmaLoadOp);
  }
}

class TXLGPUPipelinePass : public impl::TXLGPUPipelineBase<TXLGPUPipelinePass> {

public:
  using impl::TXLGPUPipelineBase<TXLGPUPipelinePass>::TXLGPUPipelineBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }


  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    OpBuilder builder(context);

    int numWarps = cast<IntegerAttr>(m->getAttr("ttg.num-warps")).getInt();
    int totalNumWarps = numWarps * numWarpgroups;
    m->setAttr("ttg.total-num-warps", builder.getI32IntegerAttr(totalNumWarps));

    lowerSmemAllocs(m);
    lowerMbars(m);
    lowerTmaLoads(m);

    pipelineWgmma(m);

    // schedule the waits
    mlir::triton::txlgpu::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    {
      SmallVector<scf::ForOp> loops;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp) > 1)
          loops.push_back(forOp);
      });

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineTMAStores(forOp);
      }

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineMMAWithScaledAcc(forOp);
      }
    }

    LLVM_DEBUG({
      LDBG("SoftwarePipeliner internal IR Dump \n");
      m.dump();
      LDBG("DONE\n\n\n");
    });

  }
};

} // namespace txlgpu
} // namespace triton
} // namespace mlir
