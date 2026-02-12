#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Attributes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"

#include "TXLGPUToLLVM/TXLGPUToLLVMPass.h"

#include "Dialect/TXLGPU/IR/Dialect.h"
#include "NVGPUToLLVM/NVGPUToLLVMPass.h" // rewriteAsPtxAsm
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "TXLGPUToLLVM/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttn = mlir::triton::nvgpu;
namespace ttx = mlir::triton::txlgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

const std::string Canonical_Warpgroup_Id_Op =
    "{\n"
    ".reg .u32 a<5>;              \n"
    "mov.u32 a0, %tid.x;          \n" // x
    "mov.u32 a1, %tid.y;          \n" // y
    "mov.u32 a2, %tid.z;          \n" // z
    "mov.u32 a3, %ntid.x;         \n" // nx
    "mov.u32 a4, %ntid.y;         \n" // ny
    "mad.lo.u32 a1, a2, a4, a1;   \n"
    "mad.lo.u32 a0, a1, a3, a0;   \n"
    "shr.u32 a0, a0, 7;           \n"
    ".reg .b32         %tmp<3>;   \n"
    "mov.u32   %tmp0, -1;         \n"
    "mov.u32   %tmp1, 31;         \n"
    "mov.u32   %tmp2, 0;          \n"
    "shfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \n"
    "}";

const std::string Lane_Id_Op =
    "{\n"
    ".reg .u32 %tid_x;              \n"
    ".reg .u32 %lane_id;            \n"
    "mov.u32 %tid_x, %tid.x;        \n"
    "rem.u32 %lane_id, %tid_x, 32;  \n" // TODO: change to $1 and add to LaneIdOp operand
    "mov.u32 $0, %lane_id;          \n"
    "}";

const std::string Warp_Id_Op =
    "{\n"
    ".reg .u32 %tid_x;              \n"
    ".reg .u32 %warp_id;            \n"
    "mov.u32 %tid_x, %tid.x;        \n"
    "shr.u32 %warp_id, %tid_x, 5;   \n" // warp_id = tid.x / 32
    "mov.u32 $0, %warp_id;          \n"
    "}";

const std::string Cluster_CTA_Rank_Op =
    "{\n"
    ".reg .u32 %ctarank;                  \n"
    "mov.u32 %ctarank, %cluster_ctarank;  \n"
    "mov.u32 $0, %ctarank;                \n"
    "}";

const std::string Reg_Alloc_Op = "setmaxnreg.inc.sync.aligned.u32 #regCount;";
const std::string Reg_Dealloc_Op = "setmaxnreg.dec.sync.aligned.u32 #regCount;";

const std::string Named_Barrier_Arrive_Op = "bar.arrive #bar, #numThreads;";
const std::string Named_Barrier_Wait_Op = "bar.sync #bar, #numThreads;";
const std::string Barrier_Wait_All_Op = "barrier.sync 0;";

template <typename SourceOp>
class TXLGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit TXLGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return ttn::rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

struct FragLocalLoadOpConversion : public ConvertOpToLLVMPattern<ttx::FragLocalLoadOp> {
public:
  FragLocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ttx::FragLocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto otherVal = op.getOther();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto fullRegTy = cast<RankedTensorType>(regVal.getType());
    auto regTy = cast<RankedTensorType>(op.getRegType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    LinearLayout regLayout = toLinearLayout(regTy);
    LinearLayout fullRegLayout = toLinearLayout(fullRegTy);

    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      cvt = getPaddedRegToSharedLayout(regLayout, paddedEnc);
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
      auto kBlock = str_attr("block");
      // NYI. We would need to emit a map.shared::cluster instruction.
      if (!cvt.isTrivialOver({kBlock})) {
        return failure();
      }
    }
    //llvm::outs() << "\n lowerFragLocalLoad\n";
    //op.dump();
    //llvm::outs() << "\n fullRegLayout\n";
    //llvm::outs() << fullRegLayout;
    //llvm::outs() << "\n regLayout\n";
    //llvm::outs() << regLayout;
    //llvm::outs() << "\n cvt\n";
    //llvm::outs() << cvt;
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
    //llvm::outs() << "\n sublayout\n";
    //llvm::outs() << cvt;
    //llvm::outs() << "\n";

    //txl: cluster
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto attr = op->getAttrOfType<IntegerAttr>("ttxg.ctaid");
    Value ctaId = Value();
    if (attr) {
        assert(attr.getType().isInteger(32) && "ttxg.ctaid must be 32 bit int\n");
        ctaId = b.i32_val(attr.getInt());
    }

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op, otherVal, op.getPred(), ctaId);

    // txl pred, broadcast only triggered in 
    //  1. normal cases, i.e. warp and lanes are full. TODO: full specified by default or fullRegLayout?
    //  2. AND, num regs is 1 and smaller than full
    //Operation *lookupPt = &rewriter.getInsertionBlock()->front();
    //int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    //int numWarps = triton::gpu::lookupNumWarps(lookupPt);
    bool needBroadcast = true;
    if (regLayout.hasInDim(kWarp)){
        int numWarps = regLayout.getInDimSize(kWarp);
        if (fullRegLayout.hasInDim(kWarp)){
            needBroadcast = needBroadcast && (numWarps == fullRegLayout.getInDimSize(kWarp));
        }
        else
            needBroadcast = false;
    }
    if (regLayout.hasInDim(kLane)){
        int numLanes = regLayout.getInDimSize(kLane);
        if (fullRegLayout.hasInDim(kLane)){
            needBroadcast = needBroadcast && (numLanes == fullRegLayout.getInDimSize(kLane));
        }
        else
            needBroadcast = false;
    }
    int numRepeats = 0;
    if (regLayout.hasInDim(kReg)){
        int numRegs = regLayout.getInDimSize(kReg);
        needBroadcast = needBroadcast && (numRegs == 1);
        if (fullRegLayout.hasInDim(kReg)){
            needBroadcast = needBroadcast && (numRegs < fullRegLayout.getInDimSize(kReg));
            numRepeats = fullRegLayout.getInDimSize(kReg) - 1;
        }
        else
            needBroadcast = false;
    }
    else
        needBroadcast = false;

    if (needBroadcast){
        for (int i = 0; i < numRepeats; i++){
            outVals.push_back(outVals[0]);
        }
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, needBroadcast?fullRegTy:regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

LogicalResult lowerFragLocalStore(Location loc, MLIRContext *ctx, Operation* op, Value regVal, Type regType,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo,
                              std::optional<MemDescType> mbarMemDescTy,
                              std::optional<SharedMemoryObject> mbarObj) {
  auto fullRegTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto regTy = cast<RankedTensorType>(regType);

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto fullRegLayout = toLinearLayout(fullRegTy);
  auto regLayout = toLinearLayout(regTy);
  auto paddedEnc =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  SmallVector<Value> newInVals;

  if (regLayout.getNumOutDims() != fullRegLayout.getNumOutDims()){
      assert(regLayout.getNumOutDims() == 1 && " currently only support 2d->1d frag smem store");
      assert(fullRegLayout.getNumOutDims() == 2 && " currently only support 2d->1d frag smem store");
      //llvm::outs() << "\n TEST: \n";
      //llvm::outs() << "\n regLayout \n";
      //llvm::outs() << regLayout;
      //llvm::outs() << "\n fullRegLayout \n";
      //llvm::outs() << fullRegLayout;

      SmallVector<StringAttr> inDimNames;
      for (auto n : fullRegLayout.getInDimNames())
          inDimNames.push_back(n);
      SmallVector<StringAttr> outDimNames;
      for (auto n : regLayout.getOutDimNames())
          outDimNames.push_back(n);

      auto subRegLayout = fullRegLayout.sublayout(inDimNames, outDimNames);
      //llvm::outs() << "\n subRegLayout \n";
      //llvm::outs() << subRegLayout;

      auto removedAction = actionRemoveBroadcastedRegs(subRegLayout);
      //llvm::outs() << "\n removedAction \n";
      //llvm::outs() << removedAction.toString();

      auto removedLayout = removedAction.apply(subRegLayout);
      //llvm::outs() << "\n removedLayout \n";
      //llvm::outs() << removedLayout;

      //llvm::outs() << "\n invals size \n";
      //llvm::outs() << inVals.size();

      //auto newInVals = removedAction.apply(inVals);
      //llvm::outs() << "\n newInVals size \n";
      //llvm::outs() << newInVals.size();
      for (auto v : removedAction.apply(inVals)){
          newInVals.push_back(v);
      }
      inVals = newInVals;

      //llvm::outs() << "\n DONE TEST \n";
  }

  if (paddedEnc) {
    cvt = getPaddedRegToSharedLayout(regLayout, paddedEnc);
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
  }
  //llvm::outs() << "\n lowerFragLocalStore\n";
  //llvm::outs() << "\n fullRegLayout\n";
  //llvm::outs() << fullRegLayout;
  //llvm::outs() << "\n regLayout\n";
  //llvm::outs() << regLayout;
  //llvm::outs() << "\n cvt\n";
  //llvm::outs() << cvt;
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  //llvm::outs() << "\n sublayout\n";
  //llvm::outs() << cvt;
  //llvm::outs() << "\n";

  //txl: cluster
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  txlgpu::FragLocalStoreOp fragLocalStore = dyn_cast<txlgpu::FragLocalStoreOp>(op);
  auto attr = op->getAttrOfType<IntegerAttr>("ttxg.ctaid");
  Value ctaId = Value();
  if (attr) {
      assert(attr.getType().isInteger(32) && "ttxg.ctaid must be 32 bit int\n");
      ctaId = b.i32_val(attr.getInt());
  }
  std::optional<Type> mbarllvmElemTy;
  if (mbarMemDescTy.has_value()) {
      if (!ctaId) {
          op->emitError("ctaid must be specified if mbar is passed");
      }
      mbarllvmElemTy = typeConverter->convertType(mbarMemDescTy.value().getElementType());
  }

  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo, op, Value(), fragLocalStore.getPred(), ctaId,
                 mbarllvmElemTy, mbarMemDescTy, mbarObj);

  return success();
}

struct FragLocalStoreOpConversion
    : public ConvertOpToLLVMPattern<ttx::FragLocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      ttx::FragLocalStoreOp>::ConvertOpToLLVMPattern;

  FragLocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<ttx::FragLocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttx::FragLocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Type regType = op.getRegType();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    Value mbar = op.getMbar();
    std::optional<MemDescType> mbarMemDescTy = std::nullopt;
    std::optional<SharedMemoryObject> mbarSmemObj = std::nullopt;
    if (mbar){
        mbarMemDescTy = cast<MemDescType>(mbar.getType());
        auto mbarElemTy = typeConverter->convertType(mbarMemDescTy.value().getElementType());
        mbarSmemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getMbar(),
                                                             mbarElemTy, rewriter);
    }
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (failed(lowerFragLocalStore(loc, ctx, op, regVal, regType, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo, mbarMemDescTy,  mbarSmemObj))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class ConvertTXLGPUToLLVM : public ConvertTXLGPUToLLVMBase<ConvertTXLGPUToLLVM> {

public:
  explicit ConvertTXLGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<TXLGPUOpGenericPattern<ttx::CanonicalWarpgroupIdOp>>(
        context, Canonical_Warpgroup_Id_Op, Constraints({"=r"}), Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::LaneIdOp>>(
        context, Lane_Id_Op, Constraints({"=r"}), Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::WarpIdOp>>(
        context, Warp_Id_Op, Constraints({"=r"}), Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::ClusterCTARankOp>>(
        context, Cluster_CTA_Rank_Op, Constraints({"=r"}), Constraints());

    patterns.add<TXLGPUOpGenericPattern<ttx::RegAllocOp>>(context, Reg_Alloc_Op, Constraints(),
                                              Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::RegDeallocOp>>(context, Reg_Dealloc_Op, Constraints(),
                                              Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::NamedBarrierArriveOp>>(
        context, Named_Barrier_Arrive_Op, Constraints(),
        Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::NamedBarrierWaitOp>>(
        context, Named_Barrier_Wait_Op, Constraints(), Constraints());
    patterns.add<TXLGPUOpGenericPattern<ttx::BarrierWaitAllOp>>(
        context, Barrier_Wait_All_Op, Constraints(), Constraints());

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTXLGPUToLLVMPass() {
  return std::make_unique<::ConvertTXLGPUToLLVM>();
}

void populateTXLGPUToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {

  patterns.add<FragLocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FragLocalStoreOpConversion>(typeConverter, targetInfo, benefit);

}

} // namespace triton
} // namespace mlir
