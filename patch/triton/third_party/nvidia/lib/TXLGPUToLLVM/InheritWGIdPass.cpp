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
#include <cstdint>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "TXLGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
namespace ttx = mlir::triton::txlgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

void addAttrAsyncIdToOpTree(Operation *op, SmallVector<int32_t> asyncIds, SpecMode mode){

  if (mode == SpecMode::WARPGROUP) {
      assert(asyncIds.size() == 1 && "asyncTaskIds for Warpgroup should always has 1 element");
  }
  std::vector<int32_t> vec{asyncIds.begin(), asyncIds.end()};

  if (op->getNumRegions() == 0) {
    // case 1: direct assign asyncId, base case
    if (mode == SpecMode::WARPGROUP) {
      setOpAttrWgId(op, asyncIds[0]);
    }
    else {
      setOpAttrWIds(op, vec);
    }
  } else {
    // case 2: ops with blocks
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<int32_t> assignedWids;
      SmallVector<int32_t> assignedElseWids;
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(ifOp, asyncIds[0]);
        assignedWids = asyncIds;
        assignedElseWids = asyncIds;
      }
      else {
        assignedWids = getOpAttrWIds(ifOp);
        assignedElseWids = getOpAttrWIds(ifOp, true);
        if (assignedWids.size() == 0) {
            setOpAttrWIds(ifOp, vec);
            assignedWids = asyncIds;
        }
        if (assignedElseWids.size() == 0) {
            setOpAttrWIds(ifOp, vec, true);
            assignedElseWids = asyncIds;
        }
      }
      for (Operation &op : ifOp.thenBlock()->getOperations()) {
          addAttrAsyncIdToOpTree(&op, assignedWids, mode);
      }
      if (ifOp.elseBlock()){
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
            addAttrAsyncIdToOpTree(&op, assignedElseWids, mode);
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(forOp, asyncIds[0]);
      }
      else {
        setOpAttrWIds(forOp, vec);
      }
      for (Operation &op : forOp.getBody()->getOperations()) {
          addAttrAsyncIdToOpTree(&op, asyncIds, mode);
      }
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      if (mode == SpecMode::WARPGROUP) {
        setOpAttrWgId(reduceOp, asyncIds[0]);
      }
      else {
        setOpAttrWIds(reduceOp, vec);
      }
      reduceOp->walk(
          [&](Operation *childOp) {
              if (isa<ReduceOp>(childOp))
                return;
              addAttrAsyncIdToOpTree(childOp, asyncIds, mode);
          });
    } else {
      llvm_unreachable("Unexpected Op with regions\n");
    }
  }

}

void addAttrAsyncIdToIfChildren(scf::IfOp ifOp, SpecMode mode){
    SmallVector<int32_t> asyncTaskIds;
    SmallVector<int32_t> elseAsyncTaskIds;
    if (mode == SpecMode::WARPGROUP) {
       auto asyncTaskId = getOpAttrWgId(ifOp);
       if (asyncTaskId != -1) {
           asyncTaskIds.push_back(asyncTaskId);
           elseAsyncTaskIds.push_back(asyncTaskId); // there is no else for isWarpgroup block
       }
    }
    else {
       auto vec = getOpAttrWIds(ifOp);
       asyncTaskIds.append(vec);
       auto elseVec = getOpAttrWIds(ifOp, true);
       elseAsyncTaskIds.append(elseVec);
    }
    if (asyncTaskIds.size() != 0) {
        for (Operation &op : ifOp.thenBlock()->getOperations()) {
            addAttrAsyncIdToOpTree(&op, asyncTaskIds, mode);
        }
    }
    if (elseAsyncTaskIds.size() != 0) {
        if (ifOp.elseBlock()){
          for (Operation &op : ifOp.elseBlock()->getOperations()) {
              addAttrAsyncIdToOpTree(&op, elseAsyncTaskIds, mode);
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
