#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/ErrorHandling.h"

#include "triton/Analysis/TXLUtility.h"

namespace mlir::triton{

void setOpAttrWgId(Operation* op, int32_t wgid){
    auto attrTy = IntegerType::get(op->getContext(), 32);
    op->setAttr("ttxg.wgid", IntegerAttr::get(attrTy, wgid));
}

int getOpAttrWgId(Operation* op){
    auto attr = op->getAttrOfType<IntegerAttr>("ttxg.wgid");
    if (attr) {
        assert(attr.getType().isInteger(32) && "ttxg.wgid must be 32 bit int\n");
        return  attr.getInt();
    }
    return -1;
}

void setOpAttrWarpReduce(Operation* op){
    op->setAttr("ttxg.warp_reduce", BoolAttr::get(op->getContext(), true));
}

bool getOpAttrWarpReduce(Operation* op){
    auto attr = op->getAttrOfType<BoolAttr>("ttxg.warp_reduce");
    if (attr) {
        return  attr.getValue();
    }
    return false;
}

void setOpRegType(Operation* op, Type type) {
    op->setAttr("ttxg.reg_type", TypeAttr::get(type));
}

Type getOpRegType(Operation* op) {
    auto attr = op->getAttrOfType<TypeAttr>("ttxg.reg_type");
    if (attr) {
        return attr.getValue();
    }
    return Type(); // returns a null Type if not set
}

}
