#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/ErrorHandling.h"

#include "txl/Dialect/TXL/IR/Dialect.h"

namespace mlir::triton{

enum SpecMode {
    WARP=0,
    WARPGROUP=1
};

bool isFakeMemoryEffects(Operation* op);

IntegerAttr getParentWithWGIDAttr(Operation *op);
IntegerAttr getParentWithWIDAttr(Operation *op);

void setOpAttrWgId(Operation* op, int32_t wgid);
int getOpAttrWgId(Operation* op);

void setOpAttrWId(Operation* op, int32_t wid);
int getOpAttrWId(Operation* op);

void setOpAttrWIds(Operation* op, std::vector<int32_t> wids);
SmallVector<int32_t> getOpAttrWIds(Operation* op);

int getExecutingThreadId(Operation * op);

void setOpAttrWarpReduce(Operation* op);

bool getOpAttrWarpReduce(Operation* op);

void setOpRegType(Operation* op, Type type);

Type getOpRegType(Operation* op);

void propagateTypeRecursively(Value &val, Type newType);
void replaceAndPropagate(Operation *srcOp, Value newValue);

Operation* getModuleFromOp(Operation *op);

std::string printModuleOp(ModuleOp &mod);

void changeForOpArgType(scf::ForOp forOp, unsigned int opNum, Type newType);

Operation* isFromTmemAlloc(Value v);
void addCompletionBarrier(DotXOp op, Value barrier, Value pred);

}
