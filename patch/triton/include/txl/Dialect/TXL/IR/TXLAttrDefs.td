#ifndef TXL_ATTR_DEFS
#define TXL_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

// Attributes for LoadOp and StoreOp
def TXL_CacheModifierXAttr : I32EnumAttr<
    "CacheModifierX", "",
    [
        I32EnumAttrCase<"NONE", 1, "none">,
        I32EnumAttrCase<"CA", 2, "ca">,
        I32EnumAttrCase<"CG", 3, "cg">,
        I32EnumAttrCase<"WB", 4, "wb">,
        I32EnumAttrCase<"CS", 5, "cs">,
        I32EnumAttrCase<"WT", 6, "wt">,
        I32EnumAttrCase<"CV", 7, "cv">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TXL_EvictionPolicyXAttr : I32EnumAttr<
    "EvictionPolicyX", "",
    [
        I32EnumAttrCase<"NORMAL", 1, "evict_normal">,
        I32EnumAttrCase<"EVICT_FIRST", 2, "evict_first">,
        I32EnumAttrCase<"EVICT_LAST", 3, "evict_last">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_PaddingOptionXAttr : I32EnumAttr<
    "PaddingOptionX", "",
    [
        I32EnumAttrCase<"PAD_ZERO", 1, "zero">,
        // We can not set the string value to "NAN" because it is a keyword in C++
        I32EnumAttrCase<"PAD_NAN", 2, "nan">
    ]> {
    let cppNamespace = "::mlir::triton";
}


#endif
