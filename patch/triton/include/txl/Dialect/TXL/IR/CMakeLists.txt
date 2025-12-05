set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TXLAttrDefs.td)
mlir_tablegen(TXLAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TXLAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TXLAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS TXLDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=txl)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=txl)
add_mlir_doc(TXLDialect TXLDialect dialects/ -gen-dialect-doc -dialect=txl)

set(LLVM_TARGET_DEFINITIONS TXLOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TXLOps TXLOps dialects/ -gen-op-doc)

add_public_tablegen_target(TXLTableGen)

