#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TXLGPUTaskIdPropagate : Pass<"txl-gpu-taskid-propagate", "mlir::ModuleOp"> {
  let summary = "Propagate async_task_id annotations based on dependencies";

  let description = [{
    This pass propagates the `async_task_id` annotation to the dependencies
    of any op that has it set.  This has the functional effect of partitioning
    the graph into multiple async tasks, based on the initial annotation.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"numConsumerGroups", "num-consumer-groups",
           "int32_t", /*default*/"0",
           "number of consumer warp groups for warp specialization">
  ];
}

def TXLGPUWSCodePartition: Pass<"txlgpu-warp-spec-code-partition", "mlir::ModuleOp"> {
  let summary = "TXLGPU warp specialization code partition";

  let description = "This pass generates warp specialized code baed on task id attributes.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
  let options = [
    Option<"numBuffers", "num-buffers",
           "int32_t", /*default*/"0",
           "number of buffering for producer-consumer">,
    Option<"numConsumerGroups", "num-consumer-groups",
           "int32_t", /*default*/"0",
           "number of consumer warp groups for warp specialization">,
    Option<"regDecProducer", "producer-reg-dec",
           "int32_t", /*default*/"40",
           "register decrement for producer warp group">,
    Option<"regIncConsumer", "consumer-reg-inc",
           "int32_t", /*default*/"232",
           "register indrement for consumer warp group">
  ];
}

#endif
