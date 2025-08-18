#ifndef TXLGPU_DIALECT
#define TXLGPU_DIALECT

include "mlir/IR/OpBase.td"

def TXLGPU_Dialect : Dialect {
  let name = "txlgpu";
  let cppNamespace = "::mlir::triton::txlgpu";

  let description = [{
    TXL GPU Dialect.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif
