#include <memory>

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
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>
#include <memory>

using namespace mlir::triton::gpu;
namespace mlir::triton::txlgpu {

#define GEN_PASS_DEF_TXLGPUSMEMALLOCLAYOUTCONVERSIONS
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-smem-alloc-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// cvt(get_buffer(ty1), ty2) -> get_buffer(ty2)
struct CvtGetBufferToGetBuffer
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    bool hasAsyncLoadOpUsers = false;
    for (auto user : op->getUsers()){
        if (isa<AsyncLoadOp>(user)){
            hasAsyncLoadOpUsers = true;
        }
    }
    if (!hasAsyncLoadOpUsers)
        return failure();

    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    if (auto getBufferOp = dyn_cast<GetBufferOp>(arg)) {
      if (!getBufferOp->hasOneUse())
          return failure();
      Operation* allocOp = getBufferOp.getSrc().getDefiningOp();

      if (isa<SmemAllocOp>(allocOp)){
          auto smemAllocOp = dyn_cast<SmemAllocOp>(allocOp);
          rewriter.setInsertionPoint(smemAllocOp);
          rewriter.replaceOpWithNewOp<SmemAllocOp>(smemAllocOp,
                  op->getResult(0).getType(),
                  smemAllocOp.getNumStages(),
                  smemAllocOp.getIsMutable()
              );
      }
      else if (isa<MbarAllocOp>(allocOp)){
          auto mbarAllocOp = dyn_cast<MbarAllocOp>(allocOp);
          rewriter.setInsertionPoint(mbarAllocOp);
          rewriter.replaceOpWithNewOp<MbarAllocOp>(mbarAllocOp,
                  op->getResult(0).getType(),
                  mbarAllocOp.getArrCount(),
                  mbarAllocOp.getNumStages()
              );
      }
      else {
          arg->emitError("get_buffer_op must precede with smem_alloc or mbar_alloc!\n");
      }

      rewriter.setInsertionPoint(arg);
      auto newGetBuffer = rewriter.replaceOpWithNewOp<GetBufferOp>(op, op->getResult(0).getType(),
                                               getBufferOp.getSrc(),
                                               getBufferOp.getIndex());

      getBufferOp->replaceAllUsesWith(newGetBuffer->getResults());
      //getBufferOp->erase();


      return success();
    }
    return failure();
  }
};

// cvt(frag_smem_load(ty1), ty2) -> frag_smem_load(ty2)
struct CvtFragSmemLoadToFragSmemLoad
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    if (auto fragSmemLoad = dyn_cast<FragSmemLoadOp>(arg)) {
      rewriter.setInsertionPoint(arg);
      auto newFragSmemLoad = rewriter.replaceOpWithNewOp<FragSmemLoadOp>(op, op->getResult(0).getType(),
                                               fragSmemLoad.getSrc(),
                                               fragSmemLoad.getOther(),
                                               fragSmemLoad.getRegType(),
                                               fragSmemLoad.getFullLayout()
                                               );

      fragSmemLoad->replaceAllUsesWith(newFragSmemLoad->getResults());

      return success();
    }
    return failure();
  }
};

// cvt(frag_smem_load(ty1), ty2) -> frag_smem_load(ty2)
struct CvtElemWiseFragSmemLoadToFragSmemLoad
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    SmallVector<Operation*> opsToChange;
    Operation* targetOp = nullptr;

    auto parentOp = op.getSrc().getDefiningOp();
    SmallVector<Operation*> queue;
    if (parentOp)
        queue.push_back(parentOp);
    while (queue.size()){
        auto curOp = queue.back();
        queue.pop_back();
        if (!curOp)
            continue;

        if (isa<FragSmemLoadOp>(curOp)){
            targetOp = curOp;
            break;
        }
        if (curOp->hasTrait<mlir::OpTrait::Elementwise>()){
            //opsToChange.push_back(user);
            auto newOp = curOp->getOperands()[0].getDefiningOp();
            if (newOp)
                queue.push_back(newOp);
            if (curOp->getNumOperands() > 1){
                auto newOp2 = curOp->getOperands()[1].getDefiningOp();
                if (newOp2)
                    queue.push_back(newOp2);
            }
        }
    }

    if (!targetOp)
      return failure();

    auto fragSmemLoad = dyn_cast<FragSmemLoadOp>(targetOp);

    rewriter.setInsertionPoint(fragSmemLoad);
    auto newFragSmemLoad = rewriter.replaceOpWithNewOp<FragSmemLoadOp>(op, op->getResult(0).getType(),
                                             fragSmemLoad.getSrc(),
                                             fragSmemLoad.getOther(),
                                             fragSmemLoad.getRegType(),
                                             fragSmemLoad.getFullLayout()
                                             );

    fragSmemLoad->replaceAllUsesWith(newFragSmemLoad->getResults());

    return success();
  }
};

void canonicalizeGetBufferOp(GetBufferOp getBufferOp){
    if (getBufferOp.getSrc().getType() !=  getBufferOp->getResult(0).getType()){
	Operation* allocOp = getBufferOp.getSrc().getDefiningOp();
	assert(isa<SmemAllocOp>(allocOp));
	SmemAllocOp smemAllocOp = dyn_cast<SmemAllocOp>(allocOp);

        OpBuilder builder(smemAllocOp);

        auto newAllocOp = builder.create<SmemAllocOp>(smemAllocOp->getLoc(), getBufferOp->getResult(0).getType(), 
                smemAllocOp.getNumStages(), smemAllocOp.getIsMutable());
        allocOp->replaceAllUsesWith(newAllocOp->getResults());
        allocOp->erase();
    }
}

class TXLGPUSmemAllocLayoutConversionsPass
    : public impl::TXLGPUSmemAllocLayoutConversionsBase<
          TXLGPUSmemAllocLayoutConversionsPass> {
public:
  // Cleanup convert ops.
  void smemAllocConversion() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet smemAllocPatterns(context);

    smemAllocPatterns.add<CvtGetBufferToGetBuffer>(context);
    smemAllocPatterns.add<CvtFragSmemLoadToFragSmemLoad>(context);
    smemAllocPatterns.add<CvtElemWiseFragSmemLoadToFragSmemLoad>(context);

    if (applyPatternsGreedily(m, std::move(smemAllocPatterns)).failed()) {
      signalPassFailure();
    }

    m->walk([&](GetBufferOp getBufferOp){
        canonicalizeGetBufferOp(getBufferOp);
    });

    LLVM_DEBUG({
      DBGS() << "Module after smem_alloc lowering:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    smemAllocConversion();
  }

};

} // namespace mlir::triton::txlgpu
