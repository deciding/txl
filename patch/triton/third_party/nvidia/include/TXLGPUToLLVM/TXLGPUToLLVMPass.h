#ifndef TRITON_CONVERSION_TXLGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TXLGPU_TO_LLVM_PASS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTXLGPUToLLVMPass();

std::unique_ptr<OperationPass<ModuleOp>> createInheritWGIdPass();

void populateTXLGPUToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);

} // namespace triton

} // namespace mlir

#endif
