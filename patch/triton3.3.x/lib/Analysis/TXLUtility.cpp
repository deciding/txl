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

}
