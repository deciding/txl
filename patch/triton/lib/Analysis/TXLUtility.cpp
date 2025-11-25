#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "triton/Analysis/TXLUtility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

namespace ttg = mlir::triton::gpu;

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

SmallVector<int> findWgidsRecursive(Operation *op) {
    if (!op)
        return {};
    // Case 0: op itself ---------------------------------
    auto ownWgids = op->getAttrOfType<DenseI32ArrayAttr>("ttxg.wgids");
    if (ownWgids)
        return llvm::to_vector(ownWgids.asArrayRef());

    Operation *parent = op->getParentOp();
    if (!parent)
        return {};  // reached root

    // Case 1: parent is an IfOp ---------------------------------------
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
        Region &thenR = ifOp.getThenRegion();
        Region &elseR = ifOp.getElseRegion();

        bool inThen = (op->getParentRegion() == &thenR);
        bool inElse = (!elseR.empty() &&
                       op->getParentRegion() == &elseR);

        auto thenAttr = ifOp->getAttrOfType<DenseI32ArrayAttr>("ttxg.wgids");
        auto elseAttr = ifOp->getAttrOfType<DenseI32ArrayAttr>("ttxg.other.wgids");

        // In THEN region → prefer ttxg.wgids
        if (inThen) {
            if (thenAttr)
                return llvm::to_vector(thenAttr.asArrayRef());
            // no attribute → keep recursing upward
            return findWgidsRecursive(parent);
        }

        // In ELSE region → prefer ttxg.other.wgids, fallback to ttxg.wgids
        if (inElse) {
            if (elseAttr)
                return llvm::to_vector(elseAttr.asArrayRef());
            // below only happens when if is not wargroup id if, and it has ttxg.wgids
            if (thenAttr)
                return llvm::to_vector(thenAttr.asArrayRef());
            return findWgidsRecursive(parent);
        }

        // Should not happen, but recurse safely
        llvm_unreachable("Either statement in then or else");
        //return findWgidsRecursive(parent);
    }

    // Case 2: parent is any normal op ---------------------------------
    auto direct = parent->getAttrOfType<DenseI32ArrayAttr>("ttxg.wgids");
    if (direct)
        return llvm::to_vector(direct.asArrayRef());

    // keep going upward
    return findWgidsRecursive(parent);
}

bool hasWarpgroupOperand(Value value) {
  // Base case: this value is defined by a warpgroup op
  if (auto op = value.getDefiningOp()) {
    if (isa<triton::IsWarpgroupOp>(op)) {
      return true;
    }
  }

  // For operation results, check all operands of the defining op
  if (auto op = value.getDefiningOp()) {
    for (Value operand : op->getOperands()) {
      if (hasWarpgroupOperand(operand)) {
        return true;
      }
    }
  }

  return false;
}

bool isWarpgroupIf(Operation* op) {
    if (isa<scf::IfOp>(op)){
        scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
        Value cond = ifOp.getCondition();
        if (auto defOp = cond.getDefiningOp()) {
          if (isa<triton::IsWarpgroupOp>(defOp)) {
            return true;
          }
          if (hasWarpgroupOperand(cond))
              op->emitError("is_warpgroup should always be used directly in if condition!\n");
          //assert(!hasWarpgroupOperand(cond) && "is_warpgroup should always be used directly in if condition!\n");
        }
    }
    return false;
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
    std::string key = "ttxg.wgid";
    auto attrTy = IntegerType::get(op->getContext(), 32);
    op->setAttr(key, IntegerAttr::get(attrTy, wgid));
}

int getOpAttrWgId(Operation* op){
    std::string key = "ttxg.wgid";
    auto attr = op->getAttrOfType<IntegerAttr>(key);
    if (attr) {
        assert(attr.getType().isInteger(32) && "ttxg.wgid must be 32 bit int\n");
        return  attr.getInt();
    }
    return -1;
}

void setOpAttrWgIds(Operation* op, ArrayRef<int32_t> wgids, bool other){
    std::string key = "ttxg.wgids";
    if (other)
        key = "ttxg.other.wgids";
    Builder builder(op);
    op->setAttr(key, builder.getDenseI32ArrayAttr(wgids));
}

SmallVector<int> getOpAttrWgIds(Operation* op, bool other){
    std::string key = "ttxg.wgids";
    if (other)
        key = "ttxg.other.wgids";
    auto attr = op->getAttrOfType<DenseI32ArrayAttr>(key);
    if (attr) {
        SmallVector<int32_t> vec(attr.asArrayRef().begin(), attr.asArrayRef().end());
        return  vec;
    }
    return SmallVector<int>();
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


}
