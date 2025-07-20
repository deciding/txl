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
    DefaultValuedAttr<I32Attr, "1">:$numStages,
    DefaultValuedAttr<BoolAttr, "false">:$isMutable
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TXL_MbarAllocOp : TXL_Op<"mbar_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "shared mem alloc";
  let description = [{
    Shared memory barrier allocation.
  }];
  let arguments = (ins
    I32Attr:$arrCount,
    DefaultValuedAttr<I32Attr, "1">:$numStages
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TXL_GetBufferOp : TXL_Op<"get_buffer", [Pure,
  TypesMatchWith<"src type matches result type", "src", "result",
                 "$_self">
]> {
  let summary = "get stage buffer";
  let description = [{
    Get stage buffer from smem with index
  }];
  let arguments = (ins
    TT_Tensor:$src,
    I32:$index
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src `,` $index attr-dict `:` functional-type(operands, $result)";
}

def TXL_MbarExpectOp : TXL_Op<"mbar_expect", [MemoryEffects<[MemWrite<SmemMemory>]>]> {
  let summary = "mbarrier.expect_tx";
  let description = [{
    mbarrier expectCount
  }];
  let arguments = (ins
    TT_Tensor:$mbar,
    I1:$pred,
    I32Attr:$expectCount
  );

  let assemblyFormat = "$mbar `,` $pred attr-dict `:` type($mbar) `,` type($pred)";
}

def TXL_MbarWaitOp : TXL_Op<"mbar_wait", [MemoryEffects<[MemRead<SmemMemory>, MemWrite<SmemMemory>]>]> {
  let summary = "mbarrier.try/test_wait";
  let description = [{
    mbarrier wait
  }];
  let arguments = (ins
    TT_Tensor:$mbar,
    I32:$phase
  );

  let assemblyFormat = "$mbar `,` $phase attr-dict `:` type($mbar) `,` type($phase)";
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
    TT_Tensor:$mbar,
    DefaultValuedAttr<TXL_CacheModifierXAttr, "::mlir::triton::CacheModifierX::NONE">:$cache,
    DefaultValuedAttr<TXL_EvictionPolicyXAttr, "::mlir::triton::EvictionPolicyX::NORMAL">:$evict
  );

  let assemblyFormat = [{
    $src `,` $desc `[` $indices `]` `,` $mbar
    oilist(
      `cacheModifier` `=` $cache |
      `evictionPolicy` `=` $evict
    )
    attr-dict `:` type($src) `,` qualified(type($desc)) `,` type($mbar)
  }];

}

#endif // TXL_OPS
