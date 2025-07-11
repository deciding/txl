#ifndef TXL_OPS
#define TXL_OPS

include "txl/Dialect/TXL/IR/TXLDialect.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"


//
// Op Base
//
class TXL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TXL_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
}

def TXL_SmemAllocOp : TXL_Op<"smem_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "shared mem alloc";
  let description = [{
    Shared memory allocation.
  }];
  let arguments = (ins
    DefaultValuedAttr<BoolAttr, "false">:$isMutable
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // TXL_OPS
