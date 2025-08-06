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

namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir::triton::txlgpu {

#define GEN_PASS_DEF_TXLGPUNAMEDBARRIERLOWER
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-named-barrier-lower"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct NamedBarrierArriveLower
    : public OpRewritePattern<tt::NamedBarrierArriveOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(tt::NamedBarrierArriveOp op,
                  PatternRewriter &rewriter) const override {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<ttng::NamedBarrierArriveOp>(op, op.getBar(), op.getNumThreads());
      return success();
  }
};
struct NamedBarrierWaitLower
    : public OpRewritePattern<tt::NamedBarrierWaitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(tt::NamedBarrierWaitOp op,
                  PatternRewriter &rewriter) const override {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<ttng::NamedBarrierWaitOp>(op, op.getBar(), op.getNumThreads());
      return success();
  }
};

class TXLGPUNamedBarrierLowerPass
    : public impl::TXLGPUNamedBarrierLowerBase<
          TXLGPUNamedBarrierLowerPass> {
public:
  // Cleanup convert ops.
  void lowerNamedBarrier() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<NamedBarrierArriveLower>(context);
    patterns.add<NamedBarrierWaitLower>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      DBGS() << "Module after named barrier lowering:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    lowerNamedBarrier();
  }

};

} // namespace mlir::triton::txlgpu
