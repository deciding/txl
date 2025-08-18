#include "TXLGPUToLLVM/TXLGPUToLLVMPass.h"

#include "Dialect/TXLGPU/IR/Dialect.h"
#include "NVGPUToLLVM/NVGPUToLLVMPass.h" // rewriteAsPtxAsm
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Analysis/TXLUtility.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "TXLGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
namespace ttx = mlir::triton::txlgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

void addAttrWgIdToOpTree(Operation *op, int wgid){

  if (op->getNumRegions() == 0) {
    // case 1: direct assign wgid, base case
    setOpAttrWgId(op, wgid);
  } else {
    // case 2: ops with blocks
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      setOpAttrWgId(ifOp, wgid);
      for (Operation &op : ifOp.thenBlock()->getOperations()) {
          addAttrWgIdToOpTree(&op, wgid);
      }
      if (ifOp.elseBlock()){
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
            addAttrWgIdToOpTree(&op, wgid);
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      setOpAttrWgId(forOp, wgid);
      for (Operation &op : forOp.getBody()->getOperations()) {
          addAttrWgIdToOpTree(&op, wgid);
      }
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      setOpAttrWgId(reduceOp, wgid);
      reduceOp->walk(
          [&](Operation *childOp) {
              if (isa<ReduceOp>(childOp))
                return;
              addAttrWgIdToOpTree(childOp, wgid);
          });
    } else {
      llvm_unreachable("Unexpected Op with regions\n");
    }
  }

}

void addAttrWgIdToIfChildren(scf::IfOp ifOp){
    int32_t asyncTaskId = getOpAttrWgId(ifOp);
    if (asyncTaskId != -1) {
        for (Operation &op : ifOp.thenBlock()->getOperations()) {
            addAttrWgIdToOpTree(&op, asyncTaskId);
        }
        if (ifOp.elseBlock()){
          for (Operation &op : ifOp.elseBlock()->getOperations()) {
              addAttrWgIdToOpTree(&op, asyncTaskId);
          }
        }
    }
}

class InheritWGId : public InheritWGIdBase<InheritWGId> {

public:
  explicit InheritWGId() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mod->walk([&](scf::IfOp op){
        addAttrWgIdToIfChildren(op);
    });

  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createInheritWGIdPass() {
  return std::make_unique<::InheritWGId>();
}

} // namespace triton
} // namespace mlir
