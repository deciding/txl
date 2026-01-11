#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "triton/Analysis/TXLUtility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton{

bool isFakeMemoryEffects(Operation* op){
    bool fakeInTXL =  isa<
        triton::MbarExpectOp, triton::MbarWaitOp, triton::MbarArriveOp,
        triton::NamedBarrierArriveOp, triton::NamedBarrierWaitOp,
        triton::DotXOp, // this truly affect tmem, but no need mbar
        triton::WGDotWaitOp,
        triton::AsyncLoadWaitOp,
        triton::TmaStoreOp, triton::TmaStoreWaitOp,
        triton::FenceProxyAsyncOp
      >(op);
    bool fakeInTXLGPU = false;
    return fakeInTXL || fakeInTXLGPU;
}


// txl
auto getParentWithWGIDAttr(Operation *op) -> IntegerAttr {
  while (op) {
    auto attr = op->getAttrOfType<IntegerAttr>("ttxg.wgid");
    if (attr)
      return attr;
    op = op->getParentOp();
  }
  return nullptr; // Return nullptr if no parent has the attribute
}

auto getParentWithWIDAttr(Operation *op) -> IntegerAttr {
  while (op) {
    auto attr = op->getAttrOfType<IntegerAttr>("ttxg.wid");
    if (attr)
      return attr;
    op = op->getParentOp();
  }
  return nullptr; // Return nullptr if no parent has the attribute
}


void changeForOpArgType(scf::ForOp forOp, unsigned int opNum, Type newType){
    auto initArgs = forOp.getInitArgs();

    auto numInductions = forOp.getNumInductionVars();
    auto numStepingVars = numInductions * 3;
    auto numYields = forOp.getYieldedValues().size();
    auto numOperands = forOp.getNumOperands();
    assert((numOperands == numYields + numStepingVars) && "Num operands is not correct!\n");
    auto iterArgNum = opNum - numStepingVars;
    auto bbArg = forOp.getRegionIterArgs()[iterArgNum];

    auto newArg = forOp.getBody()->insertArgument(iterArgNum+numInductions, newType, forOp->getLoc());
    bbArg.replaceAllUsesWith(newArg);

    forOp.getBody()->eraseArgument(iterArgNum+1+numInductions);
}

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

void setOpAttrWIds(Operation* op, std::vector<int32_t> wids){
    OpBuilder builder(op);
    auto attr = mlir::DenseI32ArrayAttr::get(builder.getContext(), wids);
    op->setAttr("ttxg.wids", attr);
}

SmallVector<int32_t> getOpAttrWIds(Operation* op){
  llvm::SmallVector<int32_t> result;
  auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("ttxg.wids");
  if (!attr)
    return result;

  result.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  return result;
}

void setOpAttrWId(Operation* op, int32_t wid){
    auto attrTy = IntegerType::get(op->getContext(), 32);
    op->setAttr("ttxg.wid", IntegerAttr::get(attrTy, wid));
}

int getOpAttrWId(Operation* op){
    auto attr = op->getAttrOfType<IntegerAttr>("ttxg.wid");
    if (attr) {
        assert(attr.getType().isInteger(32) && "ttxg.wid must be 32 bit int\n");
        return  attr.getInt();
    }
    return -1;
}

int getExecutingThreadId(Operation * op) {
    // txl
    int wgId = getOpAttrWgId(op);
    int executingThreadId = 0;
    if (wgId != -1) {
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::lookupNumWarps(op);
      int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      executingThreadId = wgId * numWarps * warpSize;
    }
    int wId = getOpAttrWId(op);
    if (wId != -1) {
      auto mod = op->getParentOfType<ModuleOp>();
      int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      executingThreadId = wId * warpSize;
    }
    return executingThreadId;
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

bool isCastOp(Operation* op){
    return isa<arith::TruncIOp, arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp, arith::ExtFOp,
        arith::SIToFPOp, arith::FPToSIOp, arith::FPToUIOp, arith::UIToFPOp, FpToFpOp>(op);
}

bool sameShapeAndElementType(Type a, Type b) {
  auto ra = dyn_cast<RankedTensorType>(a);
  auto rb = dyn_cast<RankedTensorType>(b);
  if (!ra || !rb)
    return false;
  return ra != rb && ra.getShape() == rb.getShape() &&
         ra.getElementType() == rb.getElementType();
}
void propagateTypeRecursively(Value &val, Type newType) {
  for (auto &use : val.getUses()) {
    Operation* user = use.getOwner();
    unsigned opNum = use.getOperandNumber();
    // For loop yield, backpropagate to input, and it must be a const
    if (isa<scf::ForOp>(user)){
        // propagate to forop
        //llvm::outs() << "\n case forop\n";
        auto forOp = dyn_cast<scf::ForOp>(user);
        Value suspiciousInput = forOp->getOperand(opNum);
        Type oldType = suspiciousInput.getType();
        changeForOpArgType(forOp, opNum, newType);
        Value userResult = forOp->getResult(opNum-3);
        userResult.setType(newType);
        //llvm::outs() << "\n done forop\n";
        auto iterArgNum = opNum - 3; // TODO only 1 induction?
        auto bbArg = forOp.getRegionIterArgs()[iterArgNum];
        propagateTypeRecursively(bbArg, newType);
        propagateTypeRecursively(userResult, newType);
        continue;
    }
    if (isa<scf::YieldOp>(user)){
        //llvm::outs() << "\n case yieldop\n";
        auto yieldOp = dyn_cast<scf::YieldOp>(user);
        Region *region = yieldOp->getParentRegion();
        Operation *parentOp = region->getParentOp();
        auto forOp = dyn_cast<scf::ForOp>(parentOp);
        //auto forOp = yieldOp->getParentOfType<scf::ForOp>();
        if (forOp){
            //llvm::outs() << "\n case yieldop forop\n";
            // propagate to forop
            Value suspiciousInput = forOp->getOperand(opNum+3);
            Type oldType = suspiciousInput.getType();
            changeForOpArgType(forOp, opNum+3, newType);
            Value userResult = forOp->getResult(opNum);
            userResult.setType(newType);
            //llvm::outs() << "\n done yieldop forop\n";
            propagateTypeRecursively(userResult, newType);

            // backpropagate to const forop init arg
            auto constOp = suspiciousInput.getDefiningOp<arith::ConstantOp>();
            if (constOp){
                auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
                auto newRankedType = dyn_cast<RankedTensorType>(newType);
                auto userRankedType = dyn_cast<RankedTensorType>(oldType);
                if (attr) {
                    OpBuilder builder(constOp);
                    auto scalarValue = attr.getSplatValue<Attribute>();
                    auto splatValue = SplatElementsAttr::get(newRankedType, scalarValue);
                    auto newConst = builder.create<arith::ConstantOp>(
                                      constOp.getLoc(), newType, splatValue);
                    forOp->setOperand(opNum + 3, newConst);
                }
            }
            else {
                //user->emitError("Type backpropagate failed: ForOp input is not const");
            }
            continue;
        }
        auto ifOp = dyn_cast<scf::IfOp>(parentOp);
        //auto ifOp = yieldOp->getParentOfType<scf::IfOp>();
        if (ifOp){
            //llvm::outs() << "\n case yieldop ifop\n";
            Value userResult = ifOp->getResult(opNum);
            userResult.setType(newType);
            //llvm::outs() << "\n done yieldop ifop\n";
            propagateTypeRecursively(userResult, newType);
            continue;
        }
    }

    if (user->getNumResults()){
        Value userResult = user->getResult(0);
        Type oldType = userResult.getType();

        // Priority 1: propagate to elemwise other
        if (user->hasTrait<mlir::OpTrait::Elementwise>()){
          if (user->getNumOperands() == 2) {
            Value otherOperand = user->getOperand(opNum == 0 ? 1 : 0);
            // Priority 1.1: if the other is const, should create a new const
            auto constOp = otherOperand.getDefiningOp<arith::ConstantOp>();
            if (constOp){
                auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
                auto newRankedType = dyn_cast<RankedTensorType>(newType);
                auto userRankedType = dyn_cast<RankedTensorType>(oldType);
                if (attr) {
                    //llvm::outs() << "\n case elemwiseop with const\n";
                    OpBuilder builder(constOp);
                    auto scalarValue = attr.getSplatValue<Attribute>();
                    auto splatValue = SplatElementsAttr::get(newRankedType, scalarValue);
                    auto newConst = builder.create<arith::ConstantOp>(
                                      constOp.getLoc(), newType, splatValue);
                    user->setOperand(opNum == 0 ? 1 : 0, newConst);
                    //userResult.setType(newType);
                    //propagateTypeRecursively(userResult, newType);
                    auto userBasedTensorTy = RankedTensorType::get(
                            userRankedType.getShape(),
                            userRankedType.getElementType(),
                            newRankedType.getEncoding());
                    userResult.setType(userBasedTensorTy);
                    //llvm::outs() << "\n done elemwiseop with const\n";
                    propagateTypeRecursively(userResult, userBasedTensorTy);
                    continue;
                }
            }
            // Priority 1.2: if the other is splat, should create a new splat, and op is not addptr
            auto splatOp = otherOperand.getDefiningOp<triton::SplatOp>();
            if (splatOp && !isa<AddPtrOp>(user)){
                //llvm::outs() << "\n case elemwiseop with splatop (not addptr)\n";
                auto newRankedType = dyn_cast<RankedTensorType>(newType);
                auto userRankedType = dyn_cast<RankedTensorType>(oldType);
                OpBuilder builder(splatOp);
                auto newSplat = builder.create<SplatOp>(
                                  splatOp.getLoc(), newType, splatOp.getSrc());
                user->setOperand(opNum == 0 ? 1 : 0, newSplat);
                //userResult.setType(newType);
                //propagateTypeRecursively(userResult, newType);
                auto userBasedTensorTy = RankedTensorType::get(
                        userRankedType.getShape(),
                        userRankedType.getElementType(),
                        newRankedType.getEncoding());
                userResult.setType(userBasedTensorTy);
                //llvm::outs() << "\n done elemwiseop with splatop (not addptr)\n";
                propagateTypeRecursively(userResult, userBasedTensorTy);
                continue;
            }
          }
        }

        // Priority 2: if same shape givein to new Type
        if (oldType != newType && sameShapeAndElementType(oldType, newType)) {
          //llvm::outs() << "\n case same shape same elemtype\n";
          userResult.setType(newType);
          // Recurse on this user's result
          //llvm::outs() << "\n done same shape same elemtype\n";
          propagateTypeRecursively(userResult, newType);
          continue;
        }

        // Misc: other specific ops
        if (auto reduceOp = dyn_cast<ReduceOp>(user)){
            //llvm::outs() << "\n case reduceop\n";
            auto userRankedType = dyn_cast<RankedTensorType>(oldType);
            auto newRankedType = dyn_cast<RankedTensorType>(newType);
            if (isa<IntegerType>(oldType) || isa<FloatType>(oldType)) {
                continue;
            }
            assert(userRankedType && newRankedType && "reduceop types must be ranked");
            auto sliceEnc = dyn_cast<ttg::SliceEncodingAttr>(userRankedType.getEncoding());
            assert(sliceEnc && "reduceop result must be sliced encoding");
            auto distributedSliceParent = dyn_cast<ttg::DistributedEncodingTrait>(newRankedType.getEncoding());
            assert(sliceEnc && "new reduceop result must be distributed sliced encoding");
            auto newSliceEnc = ttg::SliceEncodingAttr::get(user->getContext(), sliceEnc.getDim(), distributedSliceParent);
            SmallVector<int64_t> sliceShape;
            int dim = 0;
            for (auto s: newRankedType.getShape()){
                if (dim == sliceEnc.getDim())
                    continue;
                sliceShape.push_back(s);
                dim ++;
            }
            auto newSliceTy = RankedTensorType::get(sliceShape, newRankedType.getElementType(), newSliceEnc);
            userResult.setType(newSliceTy);
            // Recurse on this user's result
            //llvm::outs() << "\n done reduceop\n";
            propagateTypeRecursively(userResult, newSliceTy);
            continue;
        }

        if (auto expandDimsOp = dyn_cast<ExpandDimsOp>(user)) {
            //llvm::outs() << "\n case expandDimsOp\n";
            //expandDimsOp->setAttr("TXL", StringAttr::get(expandDimsOp.getContext(), "X"));
            //auto mod = dyn_cast<ModuleOp>(getModuleFromOp(expandDimsOp));
            //llvm::outs() << printModuleOp(mod);
            SmallVector<Type> returnTypes;
            if (failed(expandDimsOp.inferReturnTypes(
                            expandDimsOp.getContext(),
                            expandDimsOp.getLoc(),
                            expandDimsOp->getOperands(),
                            expandDimsOp->getAttrDictionary(),
                            expandDimsOp->getPropertiesStorage(),
                            expandDimsOp->getRegions(),
                            returnTypes))) {
                // handle failure
                user->emitError("Propagation: ExpandDimsOp not able to infer the result type");
            }
            assert(returnTypes.size() && "ExpandDimsOp return type must have been inferred");
            auto resType = dyn_cast<RankedTensorType>(returnTypes[0]);
            assert(resType && "ExpandDimsOp return type must be RankedTensorType");
            auto userResult = expandDimsOp.getResult();
            userResult.setType(resType);
            //llvm::outs() << "\n done expandDimsOp\n";
            propagateTypeRecursively(userResult, resType);
            continue;
        }
        if (auto broadcastOp = dyn_cast<BroadcastOp>(user)) {
            //llvm::outs() << "\n case BroadcastOp\n";
            auto newRankedType = dyn_cast<RankedTensorType>(newType);
            auto userRankedType = dyn_cast<RankedTensorType>(oldType);
            auto userBasedTensorTy = RankedTensorType::get(
                    userRankedType.getShape(),
                    userRankedType.getElementType(),
                    newRankedType.getEncoding());
            userResult.setType(userBasedTensorTy);
            //llvm::outs() << "\n done BroadcastOp\n";
            propagateTypeRecursively(userResult, userBasedTensorTy);
            continue;
        }
        if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
            //llvm::outs() << "\n case SelectOp\n";
            userResult.setType(newType); // Assume true/false values are/will be the same
            //llvm::outs() << "\n done SelectOp\n";
            propagateTypeRecursively(userResult, newType);
            continue;
        }

        // TODO: should we givein to Elementwise?
        if (user->hasTrait<mlir::OpTrait::Elementwise>()){
          // Priority 3: as add_ptr offsets, should propagate also back to ptr type
          if (auto addPtr = dyn_cast<AddPtrOp>(user)){
              auto ptr = addPtr.getPtr();
              if (auto splatPtr = ptr.getDefiningOp<SplatOp>()){
                //llvm::outs() << "\n case elemwiseop with addptrop\n";
                auto splatResult = splatPtr->getResult(0);
                auto splatResultType = dyn_cast<RankedTensorType>(splatResult.getType());
                auto ty = cast<RankedTensorType>(newType);
                auto ptrType = RankedTensorType::get(ty.getShape(), splatResultType.getElementType(), ty.getEncoding());
                splatResult.setType(ptrType);
                userResult.setType(ptrType);
                //llvm::outs() << "\n done elemwiseop with addptrop\n";
                // Recurse on this user's result
                propagateTypeRecursively(userResult, ptrType);
                //OpBuilder builder(splatPtr);
                //auto newSplatPtr = builder.create<SplatOp>(
                //                  splatPtr.getLoc(), newType, splatPtr.getSrc());
                continue;
              }
          }

          auto targetOp = val.getDefiningOp();
          //llvm::outs() << "\n Type givein to Elementwise op, which should knows the type better\n";
          //llvm::outs() << "op location\n";
          //llvm::outs() << targetOp->getLoc();
          //llvm::outs() << "\n op original type\n";
          //llvm::outs() << val.getType();

          // Priority 4: Elementwise, be carefull about cast
          auto userTensorTy = dyn_cast<RankedTensorType>(oldType);
          auto newTensorTy = dyn_cast<RankedTensorType>(newType);
          if (userTensorTy && newTensorTy && userTensorTy.getElementType() != newTensorTy.getElementType()){
            //llvm::outs() << "\n case cast maybe\n";
            auto userBasedTensorTy = RankedTensorType::get(
                    userTensorTy.getShape(),
                    userTensorTy.getElementType(),
                    //newTensorTy.getElementType(),
                    //userTensorTy.getEncoding()
                    newTensorTy.getEncoding() // NOTE: if using srcEnc, need to propagate.
                    );
            //val.setType(userBasedTensorTy);
            userResult.setType(userBasedTensorTy);
            //llvm::outs() << "\n done cast maybe\n";
            propagateTypeRecursively(userResult, userBasedTensorTy);
            //llvm::outs() << "\n op givein to type\n";
            //llvm::outs() << userBasedTensorTy;
            continue;
          }

          //llvm::outs() << "\n case fallback\n";
          // if using resultType, no need to propagate
          // TODO: if changed the val type, should also change other users?
          // Priority 5, fallback, should not change val type, instead, add convert_layout
          val.setType(oldType);
          //llvm::outs() << "\n done fallback\n";
          //llvm::outs() << oldType;
          return;
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


OpPrintingFlags getOpPrintingFlags() {
  auto printingFlags = OpPrintingFlags();
  printingFlags.enableDebugInfo();
  printingFlags.printNameLocAsPrefix(true);
  return printingFlags;
}

Operation* getModuleFromOp(Operation *op) {
  while (op && !isa<ModuleOp>(op))
    op = op->getParentOp();
  return isa<ModuleOp>(op) ? op : nullptr;
}

std::string printModuleOp(ModuleOp &mod) {
 std::string str;
 llvm::raw_string_ostream os(str);
 auto printingFlags = getOpPrintingFlags();
 mod.print(os, printingFlags);
 return str;
}

Operation* isFromTmemAlloc(Value v) {
  // Follow the chain backward until we reach a non-bypassable source.
  while (true) {
    // Block argument → cannot come from TmemAllocOp.
    if (isa<BlockArgument>(v)) {
      auto blockArg = dyn_cast<BlockArgument>(v);
      auto *parentBlock = blockArg.getOwner();
      auto parentOp = parentBlock->getParentOp();

      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          unsigned idx = blockArg.getArgNumber();
          // Only iter_args produce block args (skip induction var at position 0)
          if (idx > 0) {
              v = forOp.getInitArgs()[idx - 1];
          }
      }
    }

    Operation *op = v.getDefiningOp();
    if (!op)
      return nullptr;

    if (isa<scf::ForOp>(op)) {
      auto forOp = dyn_cast<scf::ForOp>(op);
      auto result = dyn_cast<OpResult>(v);
      assert(result && "ForOp result should be retrievable");
      unsigned resIdx = result.getResultNumber();

      v = forOp.getInitArgs()[resIdx];
      continue;
    }

    // Direct allocation source.
    if (isa<TmemAllocOp>(op) || isa<ttng::TMEMAllocOp>(op)) {
      return op;
    }

    // DotOp haven't been converted to TCGen05, thus acc not have passed through
    if (isa<DotOp>(op) || isa<DotXOp>(op)) {
      v = op->getOperand(2);
      continue;
    }

    // Bypassable ops: follow their source operand.
    if (auto getBuf = dyn_cast<GetBufferOp>(op)) {
      v = getBuf.getSrc();
      continue;
    }

    if (auto idx = dyn_cast<SmemIndexOp>(op)) {
      v = idx.getSrc();
      continue;
    }

    if (auto subslice = dyn_cast<SmemSubsliceOp>(op)) {
      v = subslice.getSrc();
      continue;
    }

    if (auto trans = dyn_cast<SmemTransOp>(op)) {
      v = trans.getSrc();
      continue;
    }

    if (auto reshape = dyn_cast<SmemReshapeOp>(op)) {
      v = reshape.getSrc();
      continue;
    }

    if (auto idx = dyn_cast<ttg::MemDescIndexOp>(op)) {
      v = idx.getSrc();
      continue;
    }

    if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
      v = subslice.getSrc();
      continue;
    }

    if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
      v = trans.getSrc();
      continue;
    }

    if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
      v = reshape.getSrc();
      continue;
    }

    if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      v = convert.getSrc();
      continue;
    }

    if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
      v = localLoad.getSrc();
      continue;
    }

    // Any other op: chain breaks.
    return nullptr;
  }
}

void addCompletionBarrier(DotXOp op, Value barrier, Value pred) {
  op.getBarrierPredsMutable().append(pred);
  op.getBarriersMutable().append(barrier);
}

}
