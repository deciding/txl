#ifndef TXL_OPS
#define TXL_OPS

include "txl/Dialect/TXL/IR/TXLDialect.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "txl/Dialect/TXL/IR/TXLAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td"
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
    I32Attr:$numStages,
    BoolAttr:$isMutable,
    OptionalAttr<SharedEncodingTrait>:$sharedEnc
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TXL_SmemLoadOp : TXL_Op<"smem_load"> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins
    Arg<TT_Tensor, "", [MemRead]>:$src,
    TypeAttr: $regType,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src attr-dict `:` type($src) `->` type($result) `(` $regType `)` }];
}

def TXL_SmemStoreOp : TXL_Op<"smem_store"> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TT_Tensor, "", [MemWrite]>:$dst,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

def TXL_RemoteSmemLoadOp : TXL_Op<"remote_smem_load"> {
  let summary = "Load a buffer from local memory into a distributed tensor in cluster";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins
    Arg<TT_Tensor, "", [MemRead]>:$src,
    I32: $ctaId,
    TypeAttr: $regType
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src `,` $ctaId attr-dict `:` type($src) `,` type($ctaId) `->` type($result) `(` $regType `)` }];
}

def TXL_RemoteSmemStoreOp : TXL_Op<"remote_smem_store"> {
  let summary = "Store a distributed tensor into a buffer in local memory in cluster";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TT_Tensor, "", [MemWrite]>:$dst,
    I32: $ctaId
  );

  let assemblyFormat = [{
    $src `,` $dst `,` $ctaId attr-dict `:` type($src) `,` type($dst) `,` type($ctaId)
  }];
}

def TXL_FragSmemLoadOp : TXL_Op<"frag_smem_load", [AttrSizedOperandSegments]> {
  let summary = "Load a distributed tensor from a buffer in local memory";

  let description = [{
    Load a distributed tensor from a buffer in local memory.
  }];
  let arguments = (ins
    Arg<TT_Tensor, "", [MemRead]>:$src,
    Optional<TT_Type>:$other,
    Optional<I1>:$pred,
    TypeAttr: $regType,
    BoolAttr: $fullLayout,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src (`other` $other^)? (`pred` $pred^)? attr-dict `:` functional-type(operands, $result) `(` $regType `)`}];
}

def TXL_FragSmemStoreOp : TXL_Op<"frag_smem_store", [AttrSizedOperandSegments]> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TT_Tensor, "", [MemWrite]>:$dst,
    Optional<TT_Tensor>:$mbar,
    Optional<I1>:$pred,
    TypeAttr: $regType,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );

  let assemblyFormat = [{
    $src `,` $dst (`mbar` $mbar^)? (`pred` $pred^)? attr-dict `:`
    type($src) `(` $regType `)` `,` type($dst) (`,` type($mbar)^)? (`,` type($pred)^)?
  }];
}

def TXL_TmemAllocOp : TXL_Op<"tmem_allocx", [
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
                            ]> {
  let summary = "allocate tensor memory";
  let description = [{
    This operation allocates buffer in tensor memory and return a descriptor
    containing the address and a view of the buffer.
    This is similar to ttg.local_alloc except the buffer is allocated in tensor memory.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];

  let arguments = (ins
    I32Attr:$numStages,
    BoolAttr:$isMutable,
    OptionalAttr<TTG_TensorMemoryEncodingAttr>:$sharedEnc,
    OptionalAttr<DistributedEncodingTrait>:$distributedEnc
  );

  let results = (outs
    TTG_TensorOrMemDesc:$result
  );

  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];

}

def TXL_TmemLoadOp : TXL_Op<"tmem_loadx"> {
  let summary = "Load a buffer from tensor memory into a distributed tensor";

  let description = [{
    This is similar to ttg.local_load except the result layout is restricted to only few possibility.
    Therefore we cannot combine this op with any convert layout like local_load.

    This operation takes and produces an optional token to indicate TMEM read
    on its source operand. When the tokens are present, they can
    be used to check aliasing and modref on the TMEM buffer.
  }];
  let arguments = (ins
    Arg<TTG_TensorOrMemDesc, "", [MemRead]>:$src,
    TypeAttr: $regType,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src attr-dict `:` type($src) `->` type($result) `(` $regType `)` }];

}

def TXL_TmemStoreOp : TXL_Op<"tmem_storex"> {
  let summary = "Store a distributed tensor into a buffer in tensor memory";

  let description = [{
    This is similar to ttg.local_store except the source layout is restricted to only few possibility.

    This operation takes and produces an optional token to indicate TMEM write
    on its source operand. When the tokens are present, they can
    be used to check aliasing and modref on the TMEM buffer.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TTG_TensorOrMemDesc, "", [MemWrite]>:$dst,
    DefaultValuedAttr<I32Attr, "-1">: $ctaId
  );

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

def TXL_RelayoutOp : TXL_Op<"relayout",
                                 [Pure]> {
  let summary = "convert layout";

  let arguments = (ins
    TT_Tensor:$src,
    TypeAttr: $regType
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result) `(` $regType `)`";
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

def TXL_MbarArriveOp : TXL_Op<"mbar_arrive", [AttrSizedOperandSegments,
                                                      MemoryEffects<[MemWrite<SmemMemory>]>]> {
  let summary = "mbarrier arrive";

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

    mbar_arrive %0 {trackAsyncOp = false} : !tt.ptr<i64>

  }];

  let arguments = (ins TTG_TensorOrMemDesc:$mbarrier,
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

def TXL_FenceProxyAsyncOp : TXL_Op<"fence_proxy_async", [MemoryEffects<[MemWrite]>]> {
  let summary = "fence proxy async and general";
  let description = [{
    fence proxy async
  }];

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
    TT_Tensor:$mbar,
    TT_TensorDescType:$desc,
    Variadic<I32>:$indices,
    DefaultValuedAttr<TXL_CacheModifierXAttr, "::mlir::triton::CacheModifierX::NONE">:$cache,
    DefaultValuedAttr<TXL_EvictionPolicyXAttr, "::mlir::triton::EvictionPolicyX::NORMAL">:$evict,
    DefaultValuedAttr<I32Attr, "-1">: $contiguity
  );

  let assemblyFormat = [{
    $src `,` $mbar `,` $desc `[` $indices `]`
    oilist(
      `cacheModifier` `=` $cache |
      `evictionPolicy` `=` $evict
    )
    attr-dict `:` type($src) `,` type($mbar) `,` qualified(type($desc))
  }];

}

def TXL_TmaStoreOp : TXL_Op<"tma_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "store value based on descriptor";
  let description = [{
    This operation will be lowered to Nvidia TMA store operation on targets supporting it.
    `desc` is a tensor descriptor object.
    The shape and types of `src` must match the descriptor otherwise the result is undefined.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    TT_TensorDescType:$desc,
    Variadic<I32>:$indices
  );

  let assemblyFormat = [{
    $src `,` $desc `[` $indices `]`
    attr-dict `:` qualified(type($desc)) `,` type($src)
  }];
}

def TXL_TmaStoreWaitOp : TXL_Op<"tma_store_wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "wait until all the inputs are read.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the read operations are done from the associated store operations.
    This is needed before the shared memory can be written to.
  }];

  let assemblyFormat = "attr-dict";
}

def TXL_NamedBarrierArriveOp : TXL_Op<"barrier_arrive", [MemoryEffects<[MemWrite]>]> {
  let summary = "named barrier arrive";

  let arguments = (ins I32Attr:$bar, I32Attr: $numThreads);

  let assemblyFormat = "attr-dict";
}

def TXL_NamedBarrierWaitOp : TXL_Op<"barrier_wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "named barrier wait";

  let arguments = (ins I32Attr:$bar, I32Attr: $numThreads);

  let assemblyFormat = "attr-dict";
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
  //TypesMatchWith<"src matches ptr type", "ptr", "src", "getPointeeType($_self)">,
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
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile,
    DefaultValuedAttr<I32Attr, "-1">: $contiguity
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

  //let arguments = (ins Variadic<TT_Tensor>:$asyncToken, I32Attr:$num);
  let arguments = (ins I32Attr:$num);

  let results = (outs TT_Tensor:$retToken);

  let assemblyFormat = "attr-dict `:` functional-type(operands, results)";

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
    TT_Tensor:$mbar,
    TT_TensorDescType:$desc,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset
  );

  let assemblyFormat = [{
    $src `,` $mbar `,` $desc `[` $x_offsets `,` $y_offset `]`
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // TMA gathers have resstrictions on the minimum size of the gather result.
    // This function verifies the result type.
    static LogicalResult verifyResultType(Operation *op, mlir::ShapedType type);
  }];
}


def TXL_SmemIndexOp : TXL_Op<"smem_index", [Pure, MemDescViewTrait]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor pointing to the `i`-th element of the
    input descriptor along the 0-th dimension.

    It doesn't affect the underlying memory.

    For example, suppose that
     - the input shape is 2x4x16xf16,
     - the output shape is 4x16xf16, and
     - index = 1.
    Then the output descriptor is equivalent to input[1], where input is the logical tensor.
  }];

  let arguments = (ins TT_Tensor:$src, I32:$index);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src `[` $index `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))}];
}

def TXL_SmemSubsliceOp : TXL_Op<"smem_subslice", [Pure, MemDescViewTrait]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor representing a subview of the logical tensor.
    It doesn't affect the underlying memory.

    For example, suppose that
     - the input shape is 32x16xf16,
     - the output shape is 8x16xf16, and
     - offsets = [2, 1].
    Then in Python syntax, the subview covers input[2:8+2, 1:16+1] where input is
    the logical tensor.

    The offsets must be larger or equal to the tile of the tensor (or zero).
  }];

  let arguments = (
    ins TT_Tensor:$src,
    DenseI64ArrayAttr:$shape,
    DenseI32ArrayAttr:$offsets
  );

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  // Render offsets inline as %src[0, 0] via a custom directive, but keep
  // the overall parse/print generated from this assemblyFormat.
  let assemblyFormat = [{
    $src  attr-dict `:` qualified(type($src))
    `->` qualified(type($result))
  }];

  let results = (outs TT_Tensor:$result);

}

def TXL_SmemTransOp : TXL_Op<"smem_trans", [Pure,
                                                  MemDescViewTrait,
                                                  TransposeOpInterface,
                                                  //InferTypeOpWithLayoutEquivalence,
                                                  SameOperandsAndResultElementType
                                           ]> {
  let summary = "transpose the descriptor";

  let description = [{
    This operation returns a new descriptor
    representing a transposed view of the buffer.
  }];

  let arguments = (
    ins TT_Tensor:$src,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  //let hasFolder = 1;
}

def TXL_SmemReshapeOp : TXL_Op<"smem_reshape", [Pure,
                                                      MemDescViewTrait,
                                                      SameOperandsAndResultElementType]> {
  let summary = "creates a descriptor for the new shape";

  let description = [{
    This operation returns a new descriptor representing a reshaped view of the underlying buffer.
    This doesn't affect the memory.
  }];

  let arguments = (
    ins TT_Tensor:$src,
    DenseI64ArrayAttr:$shape
  );


  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
}

def TXL_DotXOp : TXL_Op<"dotx", [Pure, AttrSizedOperandSegments,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             DeclareOpInterfaceMethods<DotOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. $inputPrecision describes how to exercise the TC
        when the inputs are f32. It can be one of: tf32, tf32x3, ieee.
        tf32: use TC with tf32 ops.
        tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp
        ieee: don't use TC, implement dot in software.
        If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.
    }];

    let arguments = (
      ins
      TT_FpIntTensor:$a,
      TT_FpIntTensor:$b,
      TT_FpIntTensor:$c,
      TTG_TensorOrMemDesc:$mbar,
      Optional<I1>:$useD,
      Optional<I1>:$pred,
      DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
      DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc
    );

    let results = (outs TT_FpIntTensor:$d);

    // attr-dict prints enums as integers.  To get inputPrecision printed as a
    // string, we need to specify it explicitly.
    let assemblyFormat = [{
      $a`,` $b`,` $c `,` $mbar (`useD` $useD^)? (`pred` $pred^)? (`,` `inputPrecision` `=` $inputPrecision^)? attr-dict `:`
      type($a) `*` type($b) type($mbar) `->` type($d) (`useD` type($useD)^)? (`pred` type($pred)^)?
    }];

    let hasVerifier = 1;
}

#endif // TXL_OPS
