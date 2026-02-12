#ifndef TXLGPU_OPS
#define TXLGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "TXLGPUDialect.td"
include "TXLGPUAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class TXLGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<TXLGPU_Dialect, mnemonic, traits>;

def TXLGPU_CanonicalWarpgroupIdOp : TXLGPU_Op<"canonical_warpgroup_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_WarpIdOp : TXLGPU_Op<"warp_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_LaneIdOp : TXLGPU_Op<"lane_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_ClusterCTARankOp : TXLGPU_Op<"cluster_cta_rank", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_RegAllocOp : TXLGPU_Op<"reg_alloc", []> {
  let arguments = (ins I32Attr: $regCount);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_RegDeallocOp : TXLGPU_Op<"reg_dealloc", []> {
  let arguments = (ins I32Attr: $regCount);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_NamedBarrierArriveOp : TXLGPU_Op<"bar_arrive", []> {
  let arguments = (ins I32Attr:$bar, I32Attr:$numThreads);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_NamedBarrierWaitOp : TXLGPU_Op<"bar_wait", []> {
  let arguments = (ins I32Attr:$bar, I32Attr:$numThreads);
  let assemblyFormat = "attr-dict";
}

def TXLGPU_BarrierWaitAllOp : TXLGPU_Op<"bar_wait_all", []> {
  let assemblyFormat = "attr-dict";
}

def MBarrier_ArriveTypeAttr : I32EnumAttr<"MBarriveType",
    "mbarrier arrive type, either 'normal', 'expect_tx', 'cp_async'",
    [
      I32EnumAttrCase<"normal", 0>,
      I32EnumAttrCase<"cp_async", 1>,
      I32EnumAttrCase<"expect_tx", 2>,
      I32EnumAttrCase<"remote", 3>,
    ]>{
  let cppNamespace = "::mlir::triton::txlgpu";
}

def TXLGPU_InitCtxOp : TXLGPU_Op<"init_ctx", [
    MemoryEffects<[MemRead]>,
    MemoryEffects<[MemWrite]>
]> {
  let summary = "init all  warp-level context";

  let description = [{
    init all  warp context 
  }];

  let assemblyFormat = "attr-dict";
}

def TXLGPU_RestoreCtxOp : TXLGPU_Op<"restore_ctx", [
    MemoryEffects<[MemRead]>,
    MemoryEffects<[MemWrite]>
]> {
  let summary = "Restore the current warp-level context";

  let arguments = (ins I32Attr: $asyncTaskId);

  let description = [{
    Restore the current warp context 
  }];

  let assemblyFormat = "attr-dict";
}

def TXLGPU_SaveCtxOp : TXLGPU_Op<"save_ctx", [
    MemoryEffects<[MemWrite]>
]> {
  let summary = "Save the current warp-level context";

  let arguments = (ins I32Attr: $asyncTaskId);

  let description = [{
    Restore the current warp context 
  }];

  let assemblyFormat = "attr-dict";
}

def TXLGPU_FragLocalLoadOp : TXLGPU_Op<"frag_local_load", [AttrSizedOperandSegments]> {
  let summary = "Load a distributed tensor from a buffer in local memory";

  let description = [{
    Load a distributed tensor from a buffer in local memory.
  }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead]>:$src,
    Optional<TT_Type>:$other,
    Optional<I1>:$pred,
    TypeAttr: $regType,
    BoolAttr: $fullLayout
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{$src (`other` $other^)? (`pred` $pred^)? attr-dict `:` functional-type(operands, $result) `(` $regType `)`}];
}

def TXLGPU_FragLocalStoreOp : TXLGPU_Op<"frag_local_store", [AttrSizedOperandSegments]> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TTG_MemDescType, "", [MemWrite]>:$dst,
    Optional<TTG_MemDescType>:$mbar,
    Optional<I1>:$pred,
    TypeAttr: $regType
  );

  let assemblyFormat = [{
    $src `,` $dst (`mbar` $mbar^)? (`pred` $pred^)? attr-dict `:` type($src) `(` $regType `)` `,` type($dst) (`,` type($mbar)^)? (`,` type($pred)^)?
  }];
}

#endif
