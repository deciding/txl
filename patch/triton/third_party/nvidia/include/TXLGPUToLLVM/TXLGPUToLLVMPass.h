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

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTXLGPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
