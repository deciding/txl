#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"
#include "txl/Dialect/TXL/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {

void SmemAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

void MbarAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

void TmemAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

LogicalResult
TmaGatherOp::verifyResultType(Operation *op,
                                                 mlir::ShapedType type) {
  if (type.getRank() != 2)
    return op->emitOpError("result must be a 2D tensor, but got ") << type;

  // The swizzling of TMA accesses matches that of the MMAv3 shared memory
  // layouts. However, these have minimum size requirements.
  // TODO: We can support smaller gather sizes by padding the `local_alloc` this
  // lowers to to the nearest minimum tile size.
  if (unsigned rows = type.getShape()[0]; rows < 8) {
    return op->emitOpError("gather must have at least 8 rows, but got ")
           << rows;
  }

  Type dtype = type.getElementType();
  if (dtype.getIntOrFloatBitWidth() > 32)
    return op->emitOpError("TMA dtype cannot be greater than 32 bits");

  unsigned minCols = 32 / dtype.getIntOrFloatBitWidth() * 8;
  if (unsigned cols = type.getShape()[1]; cols < minCols) {
    return op->emitOpError("gather of ")
           << dtype << " must have at least " << minCols << " columns, but got "
           << cols;
  }

  return success();
}

// async_load(ptr, splat(1), ...)        -> async_load(ptr, ...)
// async_load(ptr, splat(0), other, ...) -> other
struct CanonicalizeMaskedAsyncLoadPattern : public OpRewritePattern<AsyncLoadOp> {
  CanonicalizeMaskedAsyncLoadPattern(MLIRContext *context)
      : OpRewritePattern<AsyncLoadOp>(context, 1) {}

  LogicalResult matchAndRewrite(AsyncLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask)
      return failure();

    auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
    if (!splatMask)
      return failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<AsyncLoadOp>(
          loadOp, loadOp.getType(), loadOp.getSrc(), loadOp.getPtr(), Value(), Value(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    } else {
      // mask = splat(0)

      // If there's no "other", the value is "undef".  Perhaps we want to
      // optimize it in the future.x
      auto otherVal = loadOp.getOther();
      if (!otherVal)
        return failure();
      rewriter.replaceOp(loadOp, otherVal);
    }
    return success();
  }
};

void AsyncLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<CanonicalizeMaskedAsyncLoadPattern>(context);
}

//-- DotXOp --
LogicalResult
DotXOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<RankedTensorType>(operands[0].getType()).getEncoding();
  auto bEnc = cast<RankedTensorType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc && retEnc);
    Dialect &dialect = retEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult DotXOp::verify() {
  auto aTy = getA().getType();
  auto bTy = getB().getType();
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");
  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  if (!aEncoding && !bEncoding)
    return success();
  // Verify that the encodings are valid.
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");
  auto accTy = getC().getType();
  auto retEnc = accTy.getEncoding();
  if (!retEnc)
    return emitError("miss encoding of C operand");
  Dialect &dialect = retEnc.getDialect();
  auto interface = cast<DialectInferLayoutInterface>(&dialect);
  return interface->verifyDotOpEncodingCompatibility(getOperation(), aEncoding,
                                                     bEncoding);
}

bool DotXOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "txl/Dialect/TXL/IR/Ops.cpp.inc"
