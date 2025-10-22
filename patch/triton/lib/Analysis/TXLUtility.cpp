#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/ErrorHandling.h"

#include "triton/Analysis/TXLUtility.h"

namespace mlir::triton{

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
    if (user->getNumResults()){
        Value userResult = user->getResult(0);
        Type oldType = userResult.getType();

        // Priority 1: if the other is const, should create a new const
        if (user->hasTrait<mlir::OpTrait::Elementwise>()){
          if (user->getNumOperands() == 2) {
            Value otherOperand = user->getOperand(opNum == 0 ? 1 : 0);
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
          }
        }

        // Priority 2: if same shape givein to new Type
        if (oldType != newType && sameShapeAndElementType(oldType, newType)) {
          userResult.setType(newType);
          // Recurse on this user's result
          propagateTypeRecursively(userResult, newType);
          continue;
        }

        // TODO: should we givein to Elementwise?
        if (user->hasTrait<mlir::OpTrait::Elementwise>()){
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

          // fallback
          auto targetOp = val.getDefiningOp();
          //llvm::outs() << "\n Type givein to Elementwise op, which should knows the type better\n";
          //llvm::outs() << "op location\n";
          //llvm::outs() << targetOp->getLoc();
          //llvm::outs() << "\n op original type\n";
          //llvm::outs() << val.getType();

          auto userTensorTy = dyn_cast<RankedTensorType>(oldType);
          auto newTensorTy = dyn_cast<RankedTensorType>(newType);
          if (userTensorTy && newTensorTy && userTensorTy.getElementType() != newTensorTy.getElementType()){
            auto userBasedTensorTy = RankedTensorType::get(
                    userTensorTy.getShape(),
                    newTensorTy.getElementType(),
                    //userTensorTy.getEncoding()
                    newTensorTy.getEncoding()
                    );
            val.setType(userBasedTensorTy);
            //llvm::outs() << "\n op givein to type\n";
            //llvm::outs() << userBasedTensorTy;
            return;
          }

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

}
