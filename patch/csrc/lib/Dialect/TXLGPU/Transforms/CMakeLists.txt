add_triton_library(TXLGPUTransforms
  WSCodePartition.cpp

  DEPENDS
  TXLGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonToTritonGPU
  MLIRTransformUtils
)
