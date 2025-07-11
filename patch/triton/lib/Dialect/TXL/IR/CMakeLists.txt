add_triton_library(TXLIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonTableGen
  TXLTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
