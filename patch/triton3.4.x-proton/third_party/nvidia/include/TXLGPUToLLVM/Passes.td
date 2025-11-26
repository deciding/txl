#ifndef TXLGPU_CONVERSION_PASSES
#define TXLGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def InheritWGId : Pass<"inherit-wg-id", "mlir::ModuleOp"> {
    let summary = "Inherit WG Id from ancestor If";
    let description = [{

    }];
    let constructor = "mlir::triton::createInheritWGIdPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::txlgpu::TXLGPUDialect"];
}

def ConvertTXLGPUToLLVM : Pass<"convert-txlgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TXLGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTXLGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::txlgpu::TXLGPUDialect"];
}

#endif // TXLGPU_CONVERSION_PASSES
