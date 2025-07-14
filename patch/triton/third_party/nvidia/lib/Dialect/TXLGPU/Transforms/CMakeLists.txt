add_triton_library(TXLGPUTransforms
  AccelerateMatmul.cpp
  OptimizeDotOperands.cpp
  SoftwarePipeliner.cpp
  SmemAllocLegalize.cpp
  SmemAllocLayoutConversions.cpp
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
