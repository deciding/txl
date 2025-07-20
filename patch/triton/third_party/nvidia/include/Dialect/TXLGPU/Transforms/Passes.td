#ifndef TXLGPU_PASSES
#define TXLGPU_PASSES

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

def TXLGPUOptimizeDotOperands : Pass<"txlgpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "fuse transpositions";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
    hardware-accelerated transpositions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"hoistLayoutConversion", "hoist-layout-conversion",
           "bool", /*default*/"true",
           "whether to move conver to dot operand earlier pass elementwise ops">
  ];
}

def TXLGPUPipeline : Pass<"txlgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Applies software pipelining to loops in the module based on number of stages.
    This may convert some load into asynchronous loads, and multi-buffer the data.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"dumpIntermediateSteps", "dump-intermediate-steps",
           "bool", /*default*/"false",
           "Dump intermediate steps">
  ];
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
   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,

       Option<"threadsPerWarp", "threads-per-warp",
              "int32_t", /*default*/"32",
              "number of threads per warp">,
       Option<"numCTAs", "num-ctas",
             "int32_t", /*default*/"1",
             "number of ctas in a cga">,
       Option<"target", "target",
             "std::string", /*default*/"\"\"",
             "the GPU target, e.g., cuda:80, hip:gfx942">
   ];
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

def TXLGPUNamedBarrierLower : Pass<"txlgpu-named-barrier-lower", "mlir::ModuleOp"> {
  let summary = "lower named barrier control to bar.sync and bar.arrive";

  let description = [{
    lower named barrier control to bar.sync and bar.arrive
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::txlgpu::TXLGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
