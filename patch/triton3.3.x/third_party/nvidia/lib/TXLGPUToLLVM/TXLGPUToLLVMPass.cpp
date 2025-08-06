#include "TXLGPUToLLVM/TXLGPUToLLVMPass.h"

#include "Dialect/TXLGPU/IR/Dialect.h"
#include "NVGPUToLLVM/NVGPUToLLVMPass.h" // rewriteAsPtxAsm
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "TXLGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
namespace ttx = mlir::triton::txlgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

const std::string Canonical_Warpgroup_Id_Op =
    "{\n"
    ".reg .u32 a<5>;              \n"
    "mov.u32 a0, %tid.x;          \n" // x
    "mov.u32 a1, %tid.y;          \n" // y
    "mov.u32 a2, %tid.z;          \n" // z
    "mov.u32 a3, %ntid.x;         \n" // nx
    "mov.u32 a4, %ntid.y;         \n" // ny
    "mad.lo.u32 a1, a2, a4, a1;   \n"
    "mad.lo.u32 a0, a1, a3, a0;   \n"
    "shr.u32 a0, a0, 7;           \n"
    ".reg .b32         %tmp<3>;   \n"
    "mov.u32   %tmp0, -1;         \n"
    "mov.u32   %tmp1, 31;         \n"
    "mov.u32   %tmp2, 0;          \n"
    "shfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \n"
    "}";

template <typename SourceOp>
class TXLGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit TXLGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return ttn::rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class ConvertTXLGPUToLLVM : public ConvertTXLGPUToLLVMBase<ConvertTXLGPUToLLVM> {

public:
  explicit ConvertTXLGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<TXLGPUOpGenericPattern<ttx::CanonicalWarpgroupIdOp>>(
        context, Canonical_Warpgroup_Id_Op, Constraints({"=r"}), Constraints());

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTXLGPUToLLVMPass() {
  return std::make_unique<::ConvertTXLGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
