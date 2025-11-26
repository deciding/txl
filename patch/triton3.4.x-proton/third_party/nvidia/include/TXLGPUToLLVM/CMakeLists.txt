set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TXLGPUToLLVM)
add_public_tablegen_target(TXLGPUConversionPassIncGen)
