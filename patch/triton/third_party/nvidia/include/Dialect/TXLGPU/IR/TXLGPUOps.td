#ifndef TXLGPU_OPS
#define TXLGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "TXLGPUDialect.td"
include "TXLGPUAttrDefs.td"

class TXLGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<TXLGPU_Dialect, mnemonic, traits>;

def TXLGPU_CanonicalWarpgroupIdOp : TXLGPU_Op<"canonical_warpgroup_id", [Pure]> {
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
  let arguments = (ins I32:$bar, I32:$numThreads);
  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def TXLGPU_NamedBarrierWaitOp : TXLGPU_Op<"bar_wait", []> {
  let arguments = (ins I32:$bar, I32:$numThreads);
  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}


#endif
