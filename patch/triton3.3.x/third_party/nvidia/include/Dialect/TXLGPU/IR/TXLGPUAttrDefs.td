#ifndef TXLGPU_ATTRDEFS
#define TXLGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "TXLGPUDialect.td"

class TXLGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TXLGPU_Dialect, name, traits, baseCppClass> {
}

#endif
