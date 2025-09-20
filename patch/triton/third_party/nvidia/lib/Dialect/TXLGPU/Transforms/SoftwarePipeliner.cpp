#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "nvidia/include/Dialect/TXLGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Analysis/TXLUtility.h" // txl

using namespace mlir;
using namespace mlir::triton::gpu;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttx = mlir::triton::txlgpu;

namespace mlir {
namespace triton {
namespace txlgpu {

#define GEN_PASS_DEF_TXLGPUPIPELINE
#include "nvidia/include/Dialect/TXLGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "txlgpu-pipeliner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static void pipelineWgmma(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    mlir::triton::txlgpu::asyncLaunchDots(forOp);
  }
  moduleOp->walk([&](tt::WGDotWaitOp waitOp) { waitOp.erase(); });
}

bool isTMALoad(Operation *op) {
  return isa<tt::TmaLoadOp>(op);
}

// smem_alloc's tma_load users
// TODO: assume all the same, if we want to reuse smem, just dealloc all and realloc
template<typename OpTy>
SmallVector<OpTy> getSmemAllocUsers(Operation *op) {
  SmallVector<OpTy> ops;
  for (auto user0 : op->getUsers()) {
    if (isa<OpTy>(user0)) {
      ops.push_back(dyn_cast<OpTy>(user0));
    }

    if (isa<tt::GetBufferOp>(user0)) {
      for (auto user : user0->getUsers()) {
        if (isa<OpTy>(user)) {
          ops.push_back(dyn_cast<OpTy>(user));
        }
      }
    }
  }
  return ops;
}

template<typename OpTy>
OpTy getSmemAllocRoot(Value &smem){
    auto op = smem.getDefiningOp<OpTy>();
    if (op)
        return op;
    auto opBuf = smem.getDefiningOp<tt::GetBufferOp>();
    if (opBuf){
        auto src = opBuf.getSrc();
        auto rootOp = src.getDefiningOp<OpTy>();
        if (rootOp)
            return rootOp;
    }
    return nullptr;
}

// NOTE: this function is from PipeliningUtility
// originally op is load ops, now is smem_alloc op
ttg::SharedEncodingTrait getSharedEncodingTXL(Operation *op) {

  // Try to use local alloc encoding if possible.
  // local_alloc(smem_alloc)
  ttg::SharedEncodingTrait localAllocEnc;
  SmallVector<ttg::LocalAllocOp> localAllocs = getSmemAllocUsers<ttg::LocalAllocOp>(op);

  if (localAllocs.size()) {
    for (ttg::LocalAllocOp localAlloc : localAllocs) {
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

  SmallVector<tt::TmaLoadOp> tmaLoads = getSmemAllocUsers<tt::TmaLoadOp>(op);
  // tma_load overwrites local_alloc enc
  if (tmaLoads.size()) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    TypedValue<tt::TensorDescType> desc;
    desc = tmaLoads[0].getDesc();
    return ttng::getEncodingFromDescriptor(op, ty, desc);
  }

  SmallVector<tt::TmaGatherOp> tmaGathers = getSmemAllocUsers<tt::TmaGatherOp>(op);
  // tma_load overwrites local_alloc enc
  if (tmaGathers.size()) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    TypedValue<tt::TensorDescType> desc;
    desc = tmaGathers[0].getDesc();
    return ttng::getEncodingFromDescriptor(op, ty, desc);
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

// Make this general Operation *, thus numStages is passed in not inferred inside
static Value createAlloc(Operation* loadOp, int numStages,
                         ttg::SharedEncodingTrait sharedEnc
                         ) {
  OpBuilder builder(loadOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(loadOp->getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());

  // TODO: any condition?
  bufferShape.insert(bufferShape.begin(), numStages);

  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);

  //builder.setInsertionPointAfter(forOp);
  //builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return alloc;
}

// Create an allocation and init the mbarriers.
Value createBarrierAlloc(tt::MbarAllocOp& mbarAllocOp) {
  ImplicitLocOpBuilder rewriter(mbarAllocOp->getLoc(), mbarAllocOp);

  int arrCount = mbarAllocOp.getArrCount();
  int numBarriers = mbarAllocOp.getNumStages();

  Value barrierAlloc =
        triton::createScalarAlloc(rewriter, rewriter.getI64Type(), numBarriers);

  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = triton::createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InitBarrierOp>(barrierView, arrCount);
  }
  return barrierAlloc;
}

// loadOp is tmaLoadOp, alloc is the LocalAllocOp from smem_alloc
Value lowerGetBufferOp(tt::GetBufferOp getBufferOp, Value newAlloc) {
  OpBuilder builder(getBufferOp);
  Location loc = getBufferOp->getLoc();

  Value buffer = triton::createSingleBufferView(builder, newAlloc,
                                                 getBufferOp.getIndex());
  return buffer;
}


void replaceSmemBufferUses(
    Operation* oldViewOp, Value newView, // replace oldViewOp uses with newView
    ttg::MemDescType allocTy) {

      OpBuilder builder(oldViewOp);
      Location loc = oldViewOp->getLoc();

      // remove redundant local_alloc
      SmallVector<ttg::LocalAllocOp> allocsToErase;
      for (Operation *user : oldViewOp->getUsers()) {
        if (auto userAllocOp = dyn_cast<ttg::LocalAllocOp>(user)) {
          if (allocTy.getEncoding() == userAllocOp.getType().getEncoding()) {
            // replace all uses of userAlloc to newView
            tt::replaceUsesAndPropagateType(builder, userAllocOp, newView);
            allocsToErase.push_back(userAllocOp);
          }
        }
      }
      for (auto allocOp : allocsToErase) {
        allocOp.erase();
      }

      // TmaLoadOp skipped local_load op wrappings
      // If there are some uses that were not local_allocs, we need to create a
      // local_load for them.
      auto users = oldViewOp->getUsers();
      SmallVector<Operation*> ops;
      for (Operation* user : users){
          ops.push_back(user);
      }
      for (Operation* user : ops){
        if (auto asyncLoadOp = dyn_cast<tt::AsyncLoadOp>(user)) {
            asyncLoadOp->setOperand(0, newView);
        }
        else if (auto tmaLoadOp = dyn_cast<tt::TmaLoadOp>(user)) {
            tmaLoadOp->setOperand(0, newView);
        }
        else if (auto tmaGatherOp = dyn_cast<tt::TmaGatherOp>(user)) {
            tmaGatherOp->setOperand(0, newView);
        }
        else if (auto smemLoadOp = dyn_cast<tt::SmemLoadOp>(user)) {
            smemLoadOp->setOperand(0, newView);
        }
        else if (auto smemStoreOp = dyn_cast<tt::SmemStoreOp>(user)) {
            smemStoreOp->setOperand(1, newView);
        }
        else if (auto fragSmemLoadOp = dyn_cast<tt::FragSmemLoadOp>(user)) {
            fragSmemLoadOp->setOperand(0, newView);
        }
        else if (auto fragSmemStoreOp = dyn_cast<tt::FragSmemStoreOp>(user)) {
            fragSmemStoreOp->setOperand(1, newView);
        }
        else {
            auto sharedLoad = builder.create<ttg::LocalLoadOp>(
                loc, oldViewOp->getResultTypes().front(), newView);
            auto result = sharedLoad->getResults();
            oldViewOp->replaceAllUsesWith(result);
        }
      }

      //oldViewOp->erase();
}

// replace oldViewOp uses with newView
void replaceMbarBufferUses(Operation* oldViewOp, Value newView) {

      OpBuilder builder(oldViewOp);
      Location loc = oldViewOp->getLoc();
      tt::replaceUsesAndPropagateType(builder, oldViewOp, newView);
      //oldViewOp->erase();
}

void lowerMbarOps(Value& mbar, bool usedByTmaLoadOp) {
    auto mbarOp = mbar.getDefiningOp();
    OpBuilder builder(mbarOp);
    for (auto user : mbarOp->getUsers()){
        if (isa<tt::MbarExpectOp>(user)){
            auto mbarExpectOp = dyn_cast<tt::MbarExpectOp>(user);
            //Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
            builder.setInsertionPoint(user);
            builder.create<ttng::BarrierExpectOp>(user->getLoc(), mbar,
                    mbarExpectOp.getExpectCount(), mbarExpectOp.getPred());
            mbarExpectOp->erase();
        }
        else if (isa<tt::MbarWaitOp>(user)){
            auto mbarWaitOp = dyn_cast<tt::MbarWaitOp>(user);
            builder.setInsertionPoint(user);
            builder.create<ttng::WaitBarrierOp>(user->getLoc(), mbar,
                                                    mbarWaitOp.getPhase());
            mbarWaitOp->erase();
        }
        else if (isa<tt::MbarArriveOp>(user)){
            auto mbarArriveOp = dyn_cast<tt::MbarArriveOp>(user);
            builder.setInsertionPoint(user);
            builder.create<tt::MbarArriveOp>(
                    user->getLoc(),
                    mbar,
                    mbarArriveOp.getPred(),
                    /*remoteCTAId*/ nullptr,
                    mbarArriveOp.getTrackAsyncOp(), // TODO: auto by tmaload/asyncload
                    mbarArriveOp.getTxCount()
                    );
            mbarArriveOp->erase();
        }
    }

}



void lowerSmemAlloc(tt::SmemAllocOp op){
    auto sharedEnc = getSharedEncodingTXL(op);
    // this local_alloc is bind with this smem_alloc
    auto alloc = createAlloc(op, op.getNumStages(), sharedEnc);
    // these loads/tma_loads binds with local_alloc
    // TODO: add loads
    auto getBufferOps = getSmemAllocUsers<tt::GetBufferOp>(op);
    assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
           "Expected MemDescType");
    auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());

    Value buffer;
    for (auto getBufferOp : getBufferOps){
        buffer = lowerGetBufferOp(getBufferOp, alloc);
        replaceSmemBufferUses(getBufferOp, buffer, allocDescType);
        getBufferOp->erase();
    }
    replaceSmemBufferUses(op, alloc, allocDescType);
    op->erase();

    // for remaining MbarAllocOp
    // create new barriers
    // lower all barrier usages
    // remove all tt barrier ops
        
}

bool hasTMALoadUsers(Operation* op){
    for (auto user : op->getUsers()){
        if (isa<tt::TmaLoadOp>(user)){
            return true;
        }
    }
    return false;
}

void lowerMbar(tt::MbarAllocOp& op) {
    // For each group calculate the size and insert the barrier after the last
    // load.
    //sizeInBytes +=
    //    loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;

    OpBuilder builder(op);

    Value barrierAlloc = createBarrierAlloc(op);

    auto getBufferOps = getSmemAllocUsers<tt::GetBufferOp>(op);

    Value buffer;
    bool usedByTmaLoadOp = false;
    for (auto getBufferOp : getBufferOps){
        usedByTmaLoadOp = hasTMALoadUsers(getBufferOp);
        buffer = lowerGetBufferOp(getBufferOp, barrierAlloc);
        replaceMbarBufferUses(getBufferOp, buffer);
        getBufferOp->erase();
        lowerMbarOps(buffer, usedByTmaLoadOp); // removed
    }

    usedByTmaLoadOp = hasTMALoadUsers(op);
    replaceMbarBufferUses(op, barrierAlloc);
    op->erase();
    lowerMbarOps(barrierAlloc, usedByTmaLoadOp);
    // Invalidate and deallocate barrier
    //builder.setInsertionPointAfter(forOp);
    //for (int i = 0; i < numBuffers; i++) {
    //  Value barrierView =
    //      triton::createSingleBufferView(builder, barrierAlloc, i);
    //  builder.create<ttng::InvalBarrierOp>(loc, barrierView);
    //}
    //builder.create<ttg::LocalDeallocOp>(loc, barrierAlloc);
}

void lowerSmemLoad(tt::SmemLoadOp& op) {
    OpBuilder builder(op);
    Value localLoad = builder.create<ttg::LocalLoadOp>(
            op->getLoc(),
            //op.getResult().getType(),
            op.getRegType(),
            op->getOperand(0) // changed to memdesc
    );
    tt::replaceUsesAndPropagateType(builder, op, localLoad);
    op->erase();
}

void lowerSmemStore(tt::SmemStoreOp& op) {
    OpBuilder builder(op);
    ttg::LocalStoreOp local_store = builder.create<ttg::LocalStoreOp>(
            op->getLoc(),
            //newSrc,
            op->getOperand(0),
            op->getOperand(1) // changed to memdesc
    );
    op->erase();
}

bool sameShapeAndElementType(Type a, Type b) {
  auto ra = dyn_cast<RankedTensorType>(a);
  auto rb = dyn_cast<RankedTensorType>(b);
  if (!ra || !rb)
    return false;
  return ra != rb && ra.getShape() == rb.getShape() &&
         ra.getElementType() == rb.getElementType();
}
void propagateTypeRecursively(Value val, Type newType) {
  for (Operation *user : val.getUsers()) {
    if (user->getNumResults()){
        Value userResult = user->getResult(0);
        Type oldType = userResult.getType();

        if (oldType != newType && sameShapeAndElementType(oldType, newType)) {
          userResult.setType(newType);
          // Recurse on this user's result
          propagateTypeRecursively(userResult, newType);
        }
    }
  }
}
void replaceAndPropagate(Operation *srcOp, Value newValue) {
  assert(srcOp->getNumResults() == 1 &&
         "Expected single-result operation");

  Value oldResult = srcOp->getResult(0);
  Type newType = newValue.getType();

  // Replace all uses of old result
  oldResult.replaceAllUsesWith(newValue);

  // Start recursive propagation from the newValue
  propagateTypeRecursively(newValue, newType);
}

void lowerFragSmemLoad(tt::FragSmemLoadOp& op) {
    OpBuilder builder(op);
    Value other = op.getOther();
    Type resultTy = op.getRegType();
    if(other){
        resultTy = op.getResult().getType();
    }
    Value frag_local_load = builder.create<ttx::FragLocalLoadOp>(
            op->getLoc(),
            resultTy,
            op->getOperand(0),
            other,
            op.getRegType(),
            op.getBroadcast()
    );
    //tt::replaceUsesAndPropagateType(builder, op, frag_local_load);
    replaceAndPropagate(op, frag_local_load);
    op->erase();
}

void lowerFragSmemStore(tt::FragSmemStoreOp& op) {
    OpBuilder builder(op);
    ttx::FragLocalStoreOp frag_local_store = builder.create<ttx::FragLocalStoreOp>(
            op->getLoc(),
            op->getOperand(0),
            op->getOperand(1), // changed to memdesc
            op.getRegType()
    );
    op->erase();
}

void lowerSmemLoadStores(ModuleOp moduleOp) {
  moduleOp->walk([&](tt::SmemLoadOp op) {
      lowerSmemLoad(op);
  });
  moduleOp->walk([&](tt::SmemStoreOp op) {
      lowerSmemStore(op);
  });
  moduleOp->walk([&](tt::FragSmemLoadOp op) {
      lowerFragSmemLoad(op);
  });
  moduleOp->walk([&](tt::FragSmemStoreOp op) {
      lowerFragSmemStore(op);
  });
}

#if 0
void replaceUsesAndPropagateBlockArgType(OpBuilder &builder,
                                               Value oldVal, Value val) {
  SmallVector<Operation *> opsToDelete;

  // Save the operand to replace / delete later (avoid iterator invalidation).
  // TODO: can we use an early_inc iterator?
  for (OpOperand &use : oldVal.getUses()) {

    Operation *user = use.getOwner();
    auto opNum = use.getOperandNumber();

    // Non-subview/trans ops will be replaced by `val`.
    if (isa<triton::gpu::MemDescTransOp, triton::gpu::MemDescSubviewOp>(user)) {
        // `subview(old_op)` is replaced by a new `subview(val)`.
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(user);
        Value newVal;
        if (auto subview = dyn_cast<triton::gpu::MemDescSubviewOp>(user)) {
          triton::gpu::MemDescType oldType = subview.getType();
          bool isMutable =
              cast<triton::gpu::MemDescType>(val.getType()).getMutableMemory();
          Type newDstType = triton::gpu::MemDescType::get(
              oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
              oldType.getMemorySpace(), isMutable);
          newVal = builder.create<triton::gpu::MemDescSubviewOp>(
              subview.getLoc(), newDstType, val, subview.getOffsets());
          newVal.getDefiningOp()->setAttrs(user->getAttrs());
        } else if (auto trans = dyn_cast<triton::gpu::MemDescTransOp>(user)) {
          newVal = builder.create<triton::gpu::MemDescTransOp>(trans.getLoc(), val,
                                                               trans.getOrder());
          newVal.getDefiningOp()->setAttrs(user->getAttrs());
        }
        assert(newVal);
        newVal.getDefiningOp()->setAttrs(user->getAttrs());
        replaceUsesAndPropagateBlockArgType(builder, user, newVal);
        opsToDelete.push_back(use.getOwner());
    }
    else if (isa<scf::ForOp>(user)) {

        auto forOp = dyn_cast<scf::ForOp>(user);
        auto initArgs = forOp.getInitArgs();

        auto numInductions = forOp.getNumInductionVars();
        auto numStepingVars = numInductions * 3;
        auto numYields = forOp.getYieldedValues().size();
        auto numOperands = forOp.getNumOperands();
        assert((numOperands == numYields + numStepingVars) && "Num operands is not correct!\n");
        auto iterArgNum = opNum - numStepingVars;
        auto bbArg = forOp.getRegionIterArgs()[iterArgNum];

        auto newArg = forOp.getBody()->insertArgument(iterArgNum+numInductions, val.getType(), forOp->getLoc());
        replaceUsesAndPropagateBlockArgType(bbArg, newArg);

        forOp.getBody()->eraseArgument(iterArgNum+1+numInductions);
    }
    else if (isa<scf::YieldOp>(user)) {
        auto parentOp = user->getParentOp();
        if (isa<scf::IfOp>(parentOp)){
            auto ifOp = dyn_cast<scf::IfOp>(parentOp);

            auto bbArg = forOp.getRegionIterArgs()[iterArgNum];
            auto newArg = forOp.getBody()->insertArgument(opNum, val.getType(), ifOp->getLoc());
            replaceUsesAndPropagateBlockArgType(bbArg, newArg);

            forOp.getBody()->eraseArgument(iterArgNum+1+numInductions);
        }
    }
    else {
        use->set(val);
        replaceUsesAndPropagateType(builder, user, val);
    }
  }

  // Perform late op erasure.
  for (Operation *op : opsToDelete)
    op->erase();
}
#endif

void replaceUsesForBlockArg(OpBuilder &builder, Operation * oldOp, Value newValue){
    for (auto& use : oldOp->getUses()){
        // use is of type OpOperand, need to get() to get Value, but have getOperandNumber()
        auto user = use.getOwner();
        auto opNum = use.getOperandNumber();
        if (isa<scf::ForOp>(user)){

            auto forOp = dyn_cast<scf::ForOp>(user);
            auto initArgs = forOp.getInitArgs();

            auto numInductions = forOp.getNumInductionVars();
            auto numStepingVars = numInductions * 3;
            auto numYields = forOp.getYieldedValues().size();
            auto numOperands = forOp.getNumOperands();
            assert((numOperands == numYields + numStepingVars) && "Num operands is not correct!\n");
            auto iterArgNum = opNum - numStepingVars;
            auto bbArg = forOp.getRegionIterArgs()[iterArgNum];

            auto newArg = forOp.getBody()->insertArgument(iterArgNum+numInductions, newValue.getType(), forOp->getLoc());
            bbArg.replaceAllUsesWith(newArg);

            forOp.getBody()->eraseArgument(iterArgNum+1+numInductions);

            //builder.setInsertionPoint(forOp);
            //auto newForOp = builder.create<scf::ForOp>(
            //    forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
            //    forOp.getStep(), newLoopArgs);
            //assert(newForOp.getRegionIterArgs().size() ==
            //       newForOp.getInitArgs().size());
            //newForOp->setAttrs(forOp->getAttrs());

            //// Replace forOp with newForOp
            //newForOp.getRegion().takeBody(forOp.getRegion());
            //for (unsigned i = 0; i < forOp.getNumResults(); ++i)
            //  forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
        }
    }
}

// replace oldViewOp uses with newView
void replaceAsyncLoadUses(Operation* asyncLoadOp, Value tok) {

      OpBuilder builder(asyncLoadOp);
      Location loc = asyncLoadOp->getLoc();
      replaceUsesForBlockArg(builder, asyncLoadOp, tok);
      tt::replaceUsesAndPropagateType(builder, asyncLoadOp, tok);
      //oldViewOp->erase();
}

void lowerAsyncLoadOp(tt::AsyncLoadOp &asyncLoad) {

    OpBuilder builder(asyncLoad);
    Location loc = asyncLoad->getLoc();
    mlir::triton::CacheModifier cache = mlir::triton::symbolizeCacheModifier(static_cast<uint32_t>(asyncLoad.getCache())).value_or(mlir::triton::CacheModifier::NONE);
    mlir::triton::EvictionPolicy evict = mlir::triton::symbolizeEvictionPolicy(static_cast<uint32_t>(asyncLoad.getEvict())).value_or(mlir::triton::EvictionPolicy::NORMAL);

    // TODO: asyncLoad.getSrc() not working, should have an intermediate op with MemDescType
    // workaround: use getOperand instead
    Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
        loc, asyncLoad.getPtr(),
        asyncLoad.getOperand(0), // view
        asyncLoad.getMask(), asyncLoad.getOther(),
        cache, evict,
        asyncLoad.getIsVolatile());

    Operation *commit =
        builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));

    //replaceAsyncLoadUses(asyncLoad, commit->getResult(0));

    asyncLoad->erase();
}
void lowerAsyncLoadWaitOp(tt::AsyncLoadWaitOp &asyncLoadWait) {

    OpBuilder builder(asyncLoadWait);
    Location loc = asyncLoadWait->getLoc();

    // TODO: asyncLoad.getSrc() not working, should have an intermediate op with MemDescType
    // workaround: use getOperand instead
    SmallVector<Value> toks;
    Operation *wait =
        builder.create<ttg::AsyncWaitOp>(loc, toks, asyncLoadWait.getNum());

    asyncLoadWait->erase();
}
void replaceForIfResultType(scf::YieldOp &yieldOp){
    auto parentOp = yieldOp->getParentOp();
    if (isa<scf::IfOp, scf::ForOp>(parentOp)){
        //auto ifOp = dyn_cast<scf::IfOp>(parentOp);
        assert((yieldOp->getNumOperands() == parentOp->getNumResults()) && "Num of yields not same as num of if/for op results\n");
        for (int idx = 0; idx < yieldOp.getNumOperands(); idx ++){
            auto opOperandType = yieldOp->getOpOperand(idx).get().getType();
            auto resType = parentOp->getResult(idx).getType();
            if (opOperandType != resType)
                parentOp->getResult(idx).setType(opOperandType);

        }
        for (auto res : parentOp->getResults()){

        }
    }
}

void lowerTmaLoadOp(tt::TmaLoadOp &tmaLoad) {

    Value desc = tmaLoad.getDesc();

    OpBuilder builder(tmaLoad);
    Location loc = tmaLoad->getLoc();

    auto pred = builder.create<arith::ConstantIntOp>(loc, 1, 1); // TODO: tma load may need pred
    auto indices = ttng::translateTMAIndices(
        builder, tmaLoad.getLoc(),
        tmaLoad.getDesc().getType().getBlockType().getEncoding(),
        tmaLoad.getIndices());
    // TODO: tmaLoad.getMbar() not working, should have an intermediate op with MemDescType
    // workaround: use getOperand instead
    builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
        tmaLoad.getLoc(), desc, indices, 
        tmaLoad.getOperand(1), tmaLoad.getOperand(0), pred);

    tmaLoad->erase();
}

void lowerTmaGatherOp(tt::TmaGatherOp &tmaGather) {

    Value desc = tmaGather.getDesc();

    OpBuilder builder(tmaGather);
    Location loc = tmaGather->getLoc();

    auto pred = builder.create<arith::ConstantIntOp>(loc, 1, 1); // TODO: tma load may need pred
                                                                 //
    // TODO: tmaGather.getMbar() not working, should have an intermediate op with MemDescType
    // workaround: use getOperand instead
    builder.create<ttng::AsyncTMAGatherOp>(
        loc, desc,
        tmaGather.getXOffsets(), tmaGather.getYOffset(),
        tmaGather.getOperand(1), tmaGather.getOperand(0), pred);

    tmaGather->erase();
}

// helper
void printAllMbar(ModuleOp moduleOp){
  SmallVector<tt::GetBufferOp> getBufferOps;
  moduleOp->walk([&](tt::GetBufferOp getBufferOp) {
      getBufferOp->dump();
      llvm::outs() << "\n";
  });
}

void lowerSmemAllocs(ModuleOp moduleOp) {
  SmallVector<tt::SmemAllocOp> smemAllocs;

  moduleOp->walk([&](tt::SmemAllocOp smemAllocOp) { smemAllocs.push_back(smemAllocOp); });
  if (smemAllocs.empty())
    return;
  for (auto smemAllocOp : smemAllocs) {
    lowerSmemAlloc(smemAllocOp);
  }
}

void lowerMbars(ModuleOp moduleOp) {
  SmallVector<tt::MbarAllocOp> mbarAllocOps;
  moduleOp->walk([&](tt::MbarAllocOp mbarAllocOp) {
      mbarAllocOps.push_back(mbarAllocOp);
  });
  for (auto mbarAllocOp : mbarAllocOps) {
      lowerMbar(mbarAllocOp);
  }
}

void lowerLoads(ModuleOp moduleOp) {

  moduleOp->walk([&](tt::AsyncLoadOp asyncLoadOp) {
      lowerAsyncLoadOp(asyncLoadOp);
  });
  moduleOp->walk([&](tt::AsyncLoadWaitOp asyncLoadWaitOp) {
      lowerAsyncLoadWaitOp(asyncLoadWaitOp);
  });
  moduleOp->walk([&](scf::YieldOp yieldOp) {
      replaceForIfResultType(yieldOp);
  });
  moduleOp->walk([&](tt::TmaLoadOp tmaLoadOp) {
      lowerTmaLoadOp(tmaLoadOp);
  });
  moduleOp->walk([&](tt::TmaGatherOp tmaGatherOp) {
      lowerTmaGatherOp(tmaGatherOp);
  });

}

class TXLGPUPipelinePass : public impl::TXLGPUPipelineBase<TXLGPUPipelinePass> {

public:
  using impl::TXLGPUPipelineBase<TXLGPUPipelinePass>::TXLGPUPipelineBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }


  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    OpBuilder builder(context);

    LLVM_DEBUG({
      LDBG("SoftwarePipeliner Before All\n");
      m.dump();
      LDBG("DONE\n\n\n");
    });
    lowerSmemAllocs(m);
    lowerMbars(m);
    lowerLoads(m);
    lowerSmemLoadStores(m);

    pipelineWgmma(m);

    // schedule the waits
    //mlir::triton::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    {
      SmallVector<scf::ForOp> loops;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp) > 1)
          loops.push_back(forOp);
      });

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineTMAStores(forOp);
      }
    }

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
