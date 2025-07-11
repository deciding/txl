set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TXLOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TXLOps TXLOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TXLDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=txl)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=txl)
add_mlir_doc(TXLDialect TXLDialect dialects/ -gen-dialect-doc -dialect=txl)

add_public_tablegen_target(TXLTableGen)
