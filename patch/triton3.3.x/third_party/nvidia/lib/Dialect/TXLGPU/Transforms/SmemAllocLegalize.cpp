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

#define GEN_PASS_DEF_TXLGPUSMEMALLOCLEGALIZE
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-smem-alloc-legalize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class AddEncodingToSmemAlloc
    : public OpRewritePattern<SmemAllocOp> {
  //using OpRewritePattern::OpRewritePattern;

  int numWarps;
  int threadsPerWarp;
  int numCTAs;
  std::string target;

public:
  // constructor with some parameters set explicitly.
  AddEncodingToSmemAlloc(mlir::MLIRContext *context, const std::string &target, int numWarps,
                           int threadsPerWarp, int numCTAs)
  : OpRewritePattern<SmemAllocOp>(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
    numCTAs(numCTAs), target(target) {}

  mlir::LogicalResult
  matchAndRewrite(SmemAllocOp op,
                  PatternRewriter &rewriter) const override {
    auto oldTy = op.getType();
    if (oldTy.getEncoding())
        return failure();

    auto context = op->getContext();
    auto shape = oldTy.getShape();
    auto eltTy = oldTy.getElementType();

    Attribute encoding = mlir::triton::gpu::getDefaultBlockedEncoding(
            context, shape, this->numWarps, this->threadsPerWarp, this->numCTAs);
    auto tensorType = RankedTensorType::get(shape, eltTy, encoding);

    auto newSmemAlloc = rewriter.replaceOpWithNewOp<SmemAllocOp>(op, tensorType, op.getNumStages(), op.getIsMutable());

    //for (auto user : newSmemAlloc->getUsers()) {
    //  if (isa<GetBufferOp>(user)) {
    //    auto getBuffer = dyn_cast<GetBufferOp>(user);
    //    rewriter.replaceOpWithNewOp<GetBufferOp>(user, tensorType, getBuffer.getSrc(), getBuffer.getIndex());
    //  }
    //}
    return success();
  }
};

class AddEncodingToGetBuffer
    : public OpRewritePattern<GetBufferOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(GetBufferOp op,
                  PatternRewriter &rewriter) const override {
    auto srcType = op.getSrc().getType();
    auto srcTensorType = dyn_cast<RankedTensorType>(srcType);
    if (!srcTensorType)
        return failure();

    auto srcEnc = srcTensorType.getEncoding();
    if (!srcEnc)
        return failure();

    auto oldTy = op.getType();
    auto oldTensorType = dyn_cast<RankedTensorType>(oldTy);
    if (oldTensorType.getEncoding())
        return failure();

    rewriter.replaceOpWithNewOp<GetBufferOp>(op, srcTensorType, op.getSrc(), op.getIndex());

    return success();
  }
};

class TXLGPUSmemAllocLegalizePass
    : public impl::TXLGPUSmemAllocLegalizeBase<
          TXLGPUSmemAllocLegalizePass> {
public:
  // important for getting the options ctor
  using impl::TXLGPUSmemAllocLegalizeBase<TXLGPUSmemAllocLegalizePass>::TXLGPUSmemAllocLegalizeBase;
  // Cleanup convert ops.
  void smemAllocConversion() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet smemAllocPatterns(context);

    smemAllocPatterns.add<AddEncodingToSmemAlloc>(context, target, numWarps, threadsPerWarp, numCTAs);
    smemAllocPatterns.add<AddEncodingToGetBuffer>(context);

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
