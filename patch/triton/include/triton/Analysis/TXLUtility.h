#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton{

void setOpAttrWgId(Operation* op, int32_t wgid);

int getOpAttrWgId(Operation* op);

void setOpAttrWarpReduce(Operation* op);

bool getOpAttrWarpReduce(Operation* op);

void setOpRegType(Operation* op, Type type);

Type getOpRegType(Operation* op);

}
