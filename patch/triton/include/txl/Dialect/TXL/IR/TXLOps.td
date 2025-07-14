#ifndef TXL_OPS
#define TXL_OPS

include "txl/Dialect/TXL/IR/TXLDialect.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "txl/Dialect/TXL/IR/TXLAttrDefs.td"
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
// Interfaces
//
def SmemMemory : Resource<"::mlir::triton::SmemMemory">;


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

def TXL_TmaLoadOp : TXL_Op<"tma_load", [MemoryEffects<[MemWrite<SmemMemory>]>]> {
  let summary = "Load from descriptor";
  let description = [{
    This operation will be lowered to Nvidia TMA load operation on targets supporting it.
    `desc` is a tensor descriptor object.
    The destination tensor type and shape must match the descriptor otherwise the result is undefined.

    This is an escape hatch and is only there for testing/experimenting.
    This op will be removed in the future.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    TT_TensorDescType:$desc,
    Variadic<I32>:$indices,
    DefaultValuedAttr<TXL_CacheModifierXAttr, "::mlir::triton::CacheModifierX::NONE">:$cache,
    DefaultValuedAttr<TXL_EvictionPolicyXAttr, "::mlir::triton::EvictionPolicyX::NORMAL">:$evict
  );

  let assemblyFormat = [{
    $src `,` $desc `[` $indices `]`
    oilist(
      `cacheModifier` `=` $cache |
      `evictionPolicy` `=` $evict
    )
    attr-dict `:` type($src) `,` qualified(type($desc))
  }];

}

#endif // TXL_OPS
