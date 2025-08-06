#ifndef TXLGPU_CONVERSION_PASSES_H
#define TXLGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/TXLGPUToLLVM/TXLGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TXLGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
