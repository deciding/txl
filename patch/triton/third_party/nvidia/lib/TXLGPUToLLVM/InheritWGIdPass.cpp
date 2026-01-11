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

void addAttrAsyncIdToOpTree(Operation *op, int asyncId, SpecMode mode){

  if (op->getNumRegions() == 0) {
    // case 1: direct assign asyncId, base case
    if (mode == SpecMode::WARPGROUP) {
      setOpAttrWgId(op, asyncId);
    }
    else {
      setOpAttrWId(op, asyncId);
    }
  } else {
    // case 2: ops with blocks
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(ifOp, asyncId);
      }
      else {
        setOpAttrWId(ifOp, asyncId);
      }
      for (Operation &op : ifOp.thenBlock()->getOperations()) {
          addAttrAsyncIdToOpTree(&op, asyncId, mode);
      }
      if (ifOp.elseBlock()){
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
            addAttrAsyncIdToOpTree(&op, asyncId, mode);
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(forOp, asyncId);
      }
      else {
        setOpAttrWId(forOp, asyncId);
      }
      for (Operation &op : forOp.getBody()->getOperations()) {
          addAttrAsyncIdToOpTree(&op, asyncId, mode);
      }
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(reduceOp, asyncId);
      }
      else {
        setOpAttrWId(reduceOp, asyncId);
      }
      reduceOp->walk(
          [&](Operation *childOp) {
              if (isa<ReduceOp>(childOp))
                return;
              addAttrAsyncIdToOpTree(childOp, asyncId, mode);
          });
    } else {
      llvm_unreachable("Unexpected Op with regions\n");
    }
  }

}

void addAttrAsyncIdToIfChildren(scf::IfOp ifOp, SpecMode mode){
    int32_t asyncTaskId = -1;
    if (mode == SpecMode::WARPGROUP) {
       asyncTaskId = getOpAttrWgId(ifOp);
    }
    else {
       asyncTaskId = getOpAttrWId(ifOp);
    }
    if (asyncTaskId != -1) {
        for (Operation &op : ifOp.thenBlock()->getOperations()) {
            addAttrAsyncIdToOpTree(&op, asyncTaskId, mode);
        }
        if (ifOp.elseBlock()){
          for (Operation &op : ifOp.elseBlock()->getOperations()) {
              addAttrAsyncIdToOpTree(&op, asyncTaskId, mode);
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
        addAttrAsyncIdToIfChildren(op, SpecMode::WARPGROUP);
        addAttrAsyncIdToIfChildren(op, SpecMode::WARP);
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
