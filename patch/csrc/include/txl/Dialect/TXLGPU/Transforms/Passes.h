#ifndef ZLANG_DIALECT_ZLANGGPU_TRANSFORMS_PASSES_H_
#define ZLANG_DIALECT_ZLANGGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace txl {
namespace gpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "txl/Dialect/TXLGPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "txl/Dialect/TXLGPU/Transforms/Passes.h.inc"

} // namespace gpu
} // namespace txl
} // namespace mlir
#endif
