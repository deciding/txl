add_triton_library(TXLGPUTransforms
  WSCodePartition.cpp

  AccelerateMatmul.cpp
  OptimizeDotOperands.cpp
  SoftwarePipeliner.cpp
  SmemAllocLegalize.cpp
  SmemAllocLayoutConversions.cpp
  WGMMAPipeline.cpp
  NamedBarrierLower.cpp
  DEPENDS
  TXLGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
