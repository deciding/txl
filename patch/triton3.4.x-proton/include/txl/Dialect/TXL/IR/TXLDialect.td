#ifndef TXL_DIALECT
#define TXL_DIALECT

include "mlir/IR/OpBase.td"

def TXL_Dialect : Dialect {
  let name = "txl";

  let cppNamespace = "::mlir::triton";

  let summary = "The Triton TXL IR in MLIR";

  let description = [{
    Triton Dialect.

    Dependent Dialects:
      * Arith:
        * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
      * Math:
        * exp, sin, cos, log, ...
      * StructuredControlFlow:
        * for, if, while, yield, condition
      * ControlFlow:
        * br, cond_br
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

}

include "triton/Dialect/Triton/IR/TritonTypes.td"


#endif // TXL_DIALECT
