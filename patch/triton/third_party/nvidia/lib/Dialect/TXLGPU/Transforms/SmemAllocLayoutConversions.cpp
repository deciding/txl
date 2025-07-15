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

// cvt(smem_alloc(ty1), ty2) -> smem_alloc(ty2)
// DEPRECATED: convert layout should only affect local load, not global load like smem_alloc
struct SmemAllocCvtToSmemAlloc
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    if (auto smem_alloc = dyn_cast<SmemAllocOp>(arg)) {
      rewriter.setInsertionPoint(arg);
      auto new_smem_alloc = rewriter.replaceOpWithNewOp<SmemAllocOp>(op, op->getResult(0).getType(),
                                               smem_alloc.getNumStages(),
                                               smem_alloc.getIsMutable());
      rewriter.replaceAllOpUsesWith(smem_alloc, new_smem_alloc);
      rewriter.eraseOp(smem_alloc);

      return success();
    }
    return failure();
  }
};

class TXLGPUSmemAllocLayoutConversionsPass
    : public impl::TXLGPUSmemAllocLayoutConversionsBase<
          TXLGPUSmemAllocLayoutConversionsPass> {
public:
  // Cleanup convert ops.
  void smemAllocConversion() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet smemAllocPatterns(context);

    smemAllocPatterns.add<SmemAllocCvtToSmemAlloc>(context);

    if (applyPatternsGreedily(m, std::move(smemAllocPatterns)).failed()) {
      signalPassFailure();
    }

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
