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

namespace mlir::triton{

IntegerAttr getParentWithWGIDAttr(Operation *op);

SmallVector<int> findWgidsRecursive(Operation *op);

bool isWarpgroupIf(Operation* op);

void setOpAttrWgId(Operation* op, int32_t wgid);

int getOpAttrWgId(Operation* op);

void setOpAttrWgIds(Operation* op, ArrayRef<int32_t> wgids, bool other=false);

SmallVector<int> getOpAttrWgIds(Operation* op, bool other=false);

void setOpAttrWarpReduce(Operation* op);

bool getOpAttrWarpReduce(Operation* op);

void setOpRegType(Operation* op, Type type);

Type getOpRegType(Operation* op);

void replaceAndPropagate(Operation *srcOp, Value newValue);

Operation* getModuleFromOp(Operation *op);

std::string printModuleOp(ModuleOp &mod);

void changeForOpArgType(scf::ForOp forOp, unsigned int opNum, Type newType);

}
