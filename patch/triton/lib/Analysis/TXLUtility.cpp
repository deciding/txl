#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    if (isa<scf::YieldOp>(user)){
        auto yieldOp = dyn_cast<scf::YieldOp>(user);
        auto forOp = yieldOp->getParentOfType<scf::ForOp>();
        if (forOp){
            Value suspiciousInput = forOp->getOperand(opNum+3);
            Type oldType = suspiciousInput.getType();
            changeForOpArgType(forOp, opNum+3, newType);
            Value userResult = forOp->getResult(opNum);
            userResult.setType(newType);
            propagateTypeRecursively(userResult, newType);

            // backpropagate to const
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
                user->emitError("Type backpropagate failed: ForOp input is not const");
            }
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
                    propagateTypeRecursively(userResult, userBasedTensorTy);
                    continue;
                }
            }
            // Priority 1.2: if the other is splat, should create a new splat, and op is not addptr
            auto splatOp = otherOperand.getDefiningOp<triton::SplatOp>();
            if (splatOp && !isa<AddPtrOp>(user)){
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
                propagateTypeRecursively(userResult, userBasedTensorTy);
                continue;
            }
          }
        }

        // Priority 2: if same shape givein to new Type
        if (oldType != newType && sameShapeAndElementType(oldType, newType)) {
          userResult.setType(newType);
          // Recurse on this user's result
          propagateTypeRecursively(userResult, newType);
          continue;
        }

        if (auto reduceOp = dyn_cast<ReduceOp>(user)){
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
            propagateTypeRecursively(userResult, newSliceTy);
            continue;
        }
        // TODO: should we givein to Elementwise?
        if (user->hasTrait<mlir::OpTrait::Elementwise>()){
          // Priority 3: as add_ptr offsets, should propagate also back to ptr type
          if (auto addPtr = dyn_cast<AddPtrOp>(user)){
              auto ptr = addPtr.getPtr();
              if (auto splatPtr = ptr.getDefiningOp<SplatOp>()){
                auto splatResult = splatPtr->getResult(0);
                auto splatResultType = dyn_cast<RankedTensorType>(splatResult.getType());
                auto ty = cast<RankedTensorType>(newType);
                auto ptrType = RankedTensorType::get(ty.getShape(), splatResultType.getElementType(), ty.getEncoding());
                splatResult.setType(ptrType);
                userResult.setType(ptrType);
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
            auto userBasedTensorTy = RankedTensorType::get(
                    userTensorTy.getShape(),
                    userTensorTy.getElementType(),
                    //newTensorTy.getElementType(),
                    //userTensorTy.getEncoding()
                    newTensorTy.getEncoding() // NOTE: if using srcEnc, need to propagate.
                    );
            //val.setType(userBasedTensorTy);
            userResult.setType(userBasedTensorTy);
            propagateTypeRecursively(userResult, userBasedTensorTy);
            //llvm::outs() << "\n op givein to type\n";
            //llvm::outs() << userBasedTensorTy;
            continue;
          }

          // if using resultType, no need to propagate
          // TODO: if changed the val type, should also change other users?
          // Priority 5, fallback, should not change val type, instead, add convert_layout
          val.setType(oldType);
          //llvm::outs() << "\n op givein to type\n";
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

    // Direct allocation source.
    if (isa<TmemAllocOp>(op) || isa<ttng::TMEMAllocOp>(op)) {
      return op;
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

}
