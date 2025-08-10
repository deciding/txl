#ifndef TXL_OPS
#define TXL_OPS

include "txl/Dialect/TXL/IR/TXLDialect.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "txl/Dialect/TXL/IR/TXLAttrDefs.td"
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

def TXL_MbarArriveOp : TXL_Op<"mbar_arrive", [AttrSizedOperandSegments, MemoryEffects<[MemWrite<SmemMemory>]>]> {
  let summary = "mbarrier.arrive";

  let description = [{
    This operation defining the arriving action for a mbarrier.
    txCount:
        An optional attribute that set tx-count. This Op will be lowered into
        mbarrier.arrive.expect_tx if the optional attribute exist.
    trackAsyncOp:
        If true, this op will be lowered into cp.async.mbarrier.arrive.noinc.
    pred:
        Only perform arrive action when pred is true.
    remoteCtaId:
        if set, perform an remote arrive action.

    Example:

    txl.mbar_arrive %0 {trackAsyncOp = false} : !tt.ptr<i64>

  }];

  let arguments = (ins TT_Tensor:$mbar,
                       Optional<I1>:$pred,
                       Optional<I32>:$remoteCtaId,
                       I1Attr: $trackAsyncOp,
                       DefaultValuedAttr<I32Attr, "0">: $txCount
                  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}


def TXL_WGDotWaitOp : TXL_Op<"dot_wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "dot wait, specially for wg dot";
  let description = [{
    wg dot wait
  }];
  let arguments = (ins
    I32Attr:$pendings
  );

  let assemblyFormat = "attr-dict";
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

def TXL_NamedBarrierArriveOp : TXL_Op<"barrier_arrive", [MemoryEffects<[MemWrite]>]> {
  let summary = "named barrier arrive";

  let arguments = (ins I32:$bar, I32: $numThreads);

  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def TXL_NamedBarrierWaitOp : TXL_Op<"barrier_wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "named barrier wait";

  let arguments = (ins I32:$bar, I32: $numThreads);

  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def TXL_IsWarpgroupOp : TXL_Op<"is_warpgroup", []> {
  let summary = "Check if is warpgroup id";
  let description = [{
    check whether the warpgroup id is as specified
  }];
  let arguments = (ins
    DenseI32ArrayAttr:$ids
  );
  let results = (outs I1:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TXL_AsyncLoadOp : TXL_Op<"async_load", [
  AttrSizedOperandSegments,
  MemoryEffects<[MemWrite]>,
  TypesMatchWith<"src matches ptr type", "ptr", "src", "getPointeeType($_self)">,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instead of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (
    ins
    TT_Tensor:$src,
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
    Optional<I1Tensor>:$mask,
    Optional<TT_Type>:$other,

    DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
    OptionalAttr<TT_PaddingOptionXAttr>:$padding,
    DefaultValuedAttr<TXL_CacheModifierXAttr, "triton::CacheModifierX::NONE">:$cache,
    DefaultValuedAttr<TXL_EvictionPolicyXAttr, "triton::EvictionPolicyX::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let results = (outs TT_Tensor:$token);

  let assemblyFormat = [{
    $src `,` $ptr (`mask` $mask^)? (`other` $other^)?
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def TXL_AsyncLoadWaitOp : TXL_Op<"async_load_wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "async wait";

  let arguments = (ins Variadic<TT_Tensor>:$asyncToken, I32Attr:$num);

  let results = (outs TT_Tensor:$retToken);

  let assemblyFormat = "$asyncToken attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TXL_TmaGatherOp : TXL_Op<"tma_gather", [MemoryEffects<[MemRead]>]> {
  let summary = "gather multiple rows from a descriptor into a single tensor";
  let description = [{
    The `tt.experimental_desciptor_gather` op will be lowered to NVIDIA TMA
    load operations on targets that support it.

    `desc_ptr` is a pointer to the TMA descriptor allocated in global memory.
    The descriptor block must have 1 row and the indices must be a 1D tensor.
    Accordingly, the result is a 2D tensor multiple rows.

    This is an escape hatch and is only there for testing/experimenting. This
    op will be removed in the future.
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TT_TensorDescType:$desc,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset
  );

  let assemblyFormat = [{
    $src `,` $desc `[` $x_offsets `,` $y_offset `]`
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // TMA gathers have resstrictions on the minimum size of the gather result.
    // This function verifies the result type.
    static LogicalResult verifyResultType(Operation *op, mlir::ShapedType type);
  }];
}

#endif // TXL_OPS
