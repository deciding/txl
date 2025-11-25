#ifndef TRITON_DIALECT_TXLGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TXLGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h.inc"
#include "nvidia/include/Dialect/TXLGPU/IR/OpsEnums.h.inc"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_ATTRDEF_CLASSES
#include "nvidia/include/Dialect/TXLGPU/IR/TXLGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "nvidia/include/Dialect/TXLGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace txlgpu {} // namespace txlgpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TXLGPU_IR_DIALECT_H_
