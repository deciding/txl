#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TXLGPUAccelerateMatmul : Pass<"txlgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TXLGPUSmemAllocLegalize : Pass<"txlgpu-smem-alloc-legalize", "mlir::ModuleOp"> {
  let summary = "smem_alloc legalize with encodings";

  let description = [{
    smem_alloc legalize for local_alloc
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TXLGPUSmemAllocLayoutConversions : Pass<"txlgpu-smem-alloc-layout-conversions", "mlir::ModuleOp"> {
  let summary = "smem_alloc layout conversions";

  let description = [{
    smem_alloc layout conversions for local_alloc
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
