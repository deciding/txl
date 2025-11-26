add_triton_library(TXLGPUIR
  Dialect.cpp

  DEPENDS
  TXLGPUTableGen
  TXLGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
