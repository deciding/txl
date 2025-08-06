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

#endif
