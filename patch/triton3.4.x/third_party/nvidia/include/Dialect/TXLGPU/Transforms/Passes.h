#ifndef TRITON_DIALECT_TXLGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TXLGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace txlgpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

} // namespace txlgpu
} // namespace triton
} // namespace mlir
#endif
