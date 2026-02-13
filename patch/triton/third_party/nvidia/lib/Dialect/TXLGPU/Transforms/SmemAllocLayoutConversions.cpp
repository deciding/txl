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
#include "triton/Analysis/TXLUtility.h" // txl
#include "llvm/Support/Casting.h"
#include <deque>
#include <memory>

using namespace mlir::triton;
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
          auto memAllocOp = dyn_cast<SmemAllocOp>(allocOp);
          rewriter.setInsertionPoint(memAllocOp);
          auto isMutableAttr = mlir::BoolAttr::get(allocOp->getContext(), memAllocOp.getIsMutable());
          auto numStagesAttr = rewriter.getI32IntegerAttr(memAllocOp.getNumStages());
          auto sharedEnc = memAllocOp.getSharedEncAttr();
          rewriter.replaceOpWithNewOp<SmemAllocOp>(memAllocOp,
              op->getResult(0).getType(),
              numStagesAttr,
              isMutableAttr,
              sharedEnc
          );
      }
      else if (isa<TmemAllocOp>(allocOp)){
          auto memAllocOp = dyn_cast<TmemAllocOp>(allocOp);
          rewriter.setInsertionPoint(memAllocOp);
          auto isMutableAttr = mlir::BoolAttr::get(allocOp->getContext(), memAllocOp.getIsMutable());
          auto numStagesAttr = rewriter.getI32IntegerAttr(memAllocOp.getNumStages());
          auto sharedEnc = memAllocOp.getSharedEncAttr();
          auto distributedEnc = memAllocOp.getDistributedEncAttr();
          rewriter.replaceOpWithNewOp<TmemAllocOp>(memAllocOp,
              op->getResult(0).getType(),
              numStagesAttr,
              isMutableAttr,
              sharedEnc,
              distributedEnc
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


      LLVM_DEBUG({
        LDBG("CvtGetBufferToGetBuffer\n");
        ModuleOp m = dyn_cast<ModuleOp>(getModuleFromOp(op));
        LDBG(printModuleOp(m));
        LDBG("DONE\n\n\n");
      });
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
                                               fragSmemLoad.getPred(),
                                               fragSmemLoad.getRegType(),
                                               fragSmemLoad.getFullLayout()
                                               );

      fragSmemLoad->replaceAllUsesWith(newFragSmemLoad->getResults());

      LLVM_DEBUG({
        LDBG("CvtFragSmemLoadToFragSmemLoad\n");
        ModuleOp m = dyn_cast<ModuleOp>(getModuleFromOp(op));
        LDBG(printModuleOp(m));
        LDBG("DONE\n\n\n");
      });
      return success();
    }
    return failure();
  }
};

// relayout -> convert_layout
struct lowerRelayout
    : public OpRewritePattern<RelayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RelayoutOp op,
                  PatternRewriter &rewriter) const override {
    //auto convertLayout = rewriter.replaceOpWithNewOp<ConvertLayoutOp>(
    auto convertLayout = rewriter.create<ConvertLayoutOp>(
        op->getLoc(),
        op.getRegType(),
        op.getSrc()
    );

    //op->replaceAllUsesWith(convertLayout->getResults());
    replaceAndPropagate(op, convertLayout);

    LLVM_DEBUG({
      LDBG("lowerRelayout\n");
      ModuleOp m = dyn_cast<ModuleOp>(getModuleFromOp(op));
      LDBG(printModuleOp(m));
      LDBG("DONE\n\n\n");
    });
    return success();
  }
};

// frag_smem_store(cvt(ty), regTy) -> frag_smem_store(cvt(regTy), regTy)
// TODO: this should be manually done
struct FragSmemStoreCvtToFragSmemStore
    : public OpRewritePattern<FragSmemStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(FragSmemStoreOp op,
                  PatternRewriter &rewriter) const override {
    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(arg)) {
      if (cvtOp->getResult(0).getType() != op.getRegType()){
        rewriter.setInsertionPoint(arg);
        auto newCvtOp = rewriter.replaceOpWithNewOp<ConvertLayoutOp>(
                cvtOp,
                op.getRegType(),
                cvtOp.getSrc()
             );

        op->setOperand(0, newCvtOp);
        //cvtOp->replaceAllUsesWith(newCvtOp->getResults());

        LLVM_DEBUG({
          LDBG("FragSmemStoreOp\n");
          ModuleOp m = dyn_cast<ModuleOp>(getModuleFromOp(op));
          LDBG(printModuleOp(m));
          LDBG("DONE\n\n\n");
        });
        return success();
      }
    }
    return failure();
  }
};

// backwards
// cvt -> elmwise -> frag_smem_load
// elmwise -> frag_smem_load(newty)
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
                                             fragSmemLoad.getPred(),
                                             fragSmemLoad.getRegType(),
                                             fragSmemLoad.getFullLayout()
                                             );

    fragSmemLoad->replaceAllUsesWith(newFragSmemLoad->getResults());
    LLVM_DEBUG({
      LDBG("CvtElemWiseFragSmemLoadToFragSmemLoad\n");
      ModuleOp m = dyn_cast<ModuleOp>(getModuleFromOp(op));
      LDBG(printModuleOp(m));
      LDBG("DONE\n\n\n");
    });

    return success();
  }
};

void canonicalizeGetBufferOp(GetBufferOp getBufferOp){
    if (getBufferOp.getSrc().getType() !=  getBufferOp->getResult(0).getType()){
	    Operation* allocOp = getBufferOp.getSrc().getDefiningOp();
        assert(isa<SmemAllocOp>(allocOp) || isa<TmemAllocOp>(allocOp));

        OpBuilder builder(allocOp);
        if (SmemAllocOp memAllocOp = dyn_cast<SmemAllocOp>(allocOp)){
          auto numStagesAttr = builder.getI32IntegerAttr(memAllocOp.getNumStages());
          auto isMutableAttr = mlir::BoolAttr::get(getBufferOp->getContext(), memAllocOp.getIsMutable());
          auto sharedEnc = memAllocOp.getSharedEncAttr();
          SmemAllocOp newAllocOp;
          newAllocOp = builder.create<SmemAllocOp>(memAllocOp->getLoc(), getBufferOp->getResult(0).getType(), 
                  numStagesAttr, isMutableAttr, sharedEnc);
          allocOp->replaceAllUsesWith(newAllocOp->getResults());
          allocOp->erase();
        }
        else if (TmemAllocOp memAllocOp = dyn_cast<TmemAllocOp>(allocOp)){
          auto numStagesAttr = builder.getI32IntegerAttr(memAllocOp.getNumStages());
          auto isMutableAttr = mlir::BoolAttr::get(getBufferOp->getContext(), memAllocOp.getIsMutable());
          auto sharedEnc = memAllocOp.getSharedEncAttr();
          auto distributedEnc = memAllocOp.getDistributedEncAttr();
          TmemAllocOp newAllocOp;
          newAllocOp = builder.create<TmemAllocOp>(memAllocOp->getLoc(), getBufferOp->getResult(0).getType(), 
                  numStagesAttr, isMutableAttr, sharedEnc, distributedEnc);
          allocOp->replaceAllUsesWith(newAllocOp->getResults());
          allocOp->erase();
        }
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
    //smemAllocPatterns.add<CvtElemWiseFragSmemLoadToFragSmemLoad>(context);
    smemAllocPatterns.add<lowerRelayout>(context);
    smemAllocPatterns.add<FragSmemStoreCvtToFragSmemStore>(context);

    if (applyPatternsGreedily(m, std::move(smemAllocPatterns)).failed()) {
      signalPassFailure();
    }

    m->walk([&](GetBufferOp getBufferOp){
        canonicalizeGetBufferOp(getBufferOp);
    });

    LLVM_DEBUG({
      LDBG("Module after smem_alloc lowering:\n");
      LDBG(printModuleOp(m));
      LDBG("DONE\n\n\n");
    });
  }

  void runOnOperation() override {
    smemAllocConversion();
  }

};

} // namespace mlir::triton::txlgpu
