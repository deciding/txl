#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace txlgpu {

#define GEN_PASS_DEF_TXLGPUPIPELINE
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-pipeliner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

bool isTMALoad(Operation *op) {
  return isa<tt::TmaLoadOp>(op);
}

// smem_alloc's tma_load users
// TODO: assume all the same, if we want to reuse smem, just dealloc all and realloc
Operation* getTMALoadUser(Operation *op) {
  // Iterate through all users of the operation
  for (auto user : op->getUsers()) {
    // Try to cast the user to TmaLoadOp
    if (isa<tt::TmaLoadOp>(user)) {
      // Return the first valid TmaLoadOp we find
      return user;
    }
  }
  // Return nullptr if no TmaLoadOp user was found
  return nullptr;
}

// originally op is load ops, now is smem_alloc op
ttg::SharedEncodingTrait getSharedEncoding(Operation *op) {
  llvm::outs() << "enter\n";

  // Try to use local alloc encoding if possible.
  // local_alloc(smem_alloc)
  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(op->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : op->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc; // get local_alloc result smem enc
      }
      if (enc != localAllocEnc) {
        // Some users have different encoding than others.
        // Use one of the encodings, and warn about the performance issue.
        op->emitRemark()
            << "Pipelining load with different use encodings. This will lead "
               "to layout conversions and performance degradation.";
        continue;
      }
    }
  }

  // smem_alloc result type
  auto opty = op->getResultTypes()[0];
  auto ty = cast<RankedTensorType>(opty);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto order = ttg::getOrder(ty);

  // tma_load overwrites local_alloc enc
  if (Operation* userOp = getTMALoadUser(op)) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    if (localAllocEnc) {
      auto sharedMMALayout =
              dyn_cast<ttg::NVMMASharedEncodingAttr>(localAllocEnc);
      if (sharedMMALayout) {
        assert(!sharedMMALayout.getFp4Padded() &&
               "TMA load for mixed precision MMAv5 is not supported yet.");
      }
    }
    auto res = ttg::NVMMASharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType(),
        /*fp4Padded*/ false);
    return res;
  }

  if (localAllocEnc)
    return localAllocEnc;

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
          .value_or(nullptr);

  if (localAllocEnc)
    return localAllocEnc;

  // Use generic layout. This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc
                         ) {
  OpBuilder builder(loadOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(loadOp->getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  //bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);

  //builder.setInsertionPointAfter(forOp);
  //builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return alloc;
}

//void createTMAAsyncLoad(tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
//                        Value insertIdx, Value extractIdx, Value barrier,
//                        Operation *waitOp, CoarseSchedule &schedule) {
//  return createTMAAsyncCopy(loadOp, loadOp.getDesc(), alloc, insertIdx,
//                            extractIdx, barrier, waitOp, schedule,
//                            [&](OpBuilderForStage &builder, Value tmaPtr,
//                                Value barrier, Value view, Value pred) {
//                              builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
//                                  loadOp.getLoc(), tmaPtr, loadOp.getIndices(),
//                                  barrier, view, pred);
//                            });
//}

void lowerSmemAlloc(SmemAllocOp op){
    auto sharedEnc = getSharedEncoding(op);
    auto alloc = createAlloc(op, sharedEnc);
}

void lowerSmemAllocs(ModuleOp moduleOp) {
  SmallVector<SmemAllocOp> smemAllocs;
  moduleOp->walk([&](SmemAllocOp smemAllocOp) { smemAllocs.push_back(smemAllocOp); });
  if (smemAllocs.empty())
    return;
  for (auto smemAllocOp : smemAllocs) {
    lowerSmemAlloc(smemAllocOp);
  }
}

class TXLGPUPipelinePass : public impl::TXLGPUPipelineBase<TXLGPUPipelinePass> {

public:
  using impl::TXLGPUPipelineBase<TXLGPUPipelinePass>::TXLGPUPipelineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    lowerSmemAllocs(m);

    LLVM_DEBUG({
      LDBG("SoftwarePipeliner internal IR Dump \n");
      m.dump();
      LDBG("DONE\n\n\n");
    });

  }
};

} // namespace txlgpu
} // namespace triton
} // namespace mlir
