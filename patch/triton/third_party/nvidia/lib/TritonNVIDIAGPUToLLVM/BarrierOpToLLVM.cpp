/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

// txl
#include "triton/Analysis/TXLUtility.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "nvidia/include/Dialect/TXLGPU/IR/Dialect.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

// txl
namespace tt = mlir::triton;
namespace ttx = mlir::triton::txlgpu;

namespace {

// txl
struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (op->hasAttr("bar_id")) {
      // llvm.nvvm.barrier0 doesn't support bar_id and num_threads attributes,
      // so we have to lower it to ptx manually.
      auto barId = op->getAttrOfType<IntegerAttr>("bar_id").getInt();
      auto numThreads = op->getAttrOfType<IntegerAttr>("num_threads").getInt();
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &barSyncOp = *ptxBuilder.create<>("bar.sync");
      barSyncOp(ptxBuilder.newConstantOperand(barId),
                ptxBuilder.newConstantOperand(numThreads));
      auto voidTy = void_ty(op->getContext());
      ptxBuilder.launch(rewriter, op->getLoc(), voidTy);
      rewriter.eraseOp(op);
      return success();
    }
    // Otherwise we let the default lowering handle it
    return failure();
  }
};

struct FenceAsyncSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = NVVM::ProxyKind::async_shared;
    auto space = op.getBCluster() ? NVVM::SharedSpace::shared_cluster
                                  : NVVM::SharedSpace::shared_cta;
    auto ctx = rewriter.getContext();
    auto spaceAttr = NVVM::SharedSpaceAttr::get(ctx, space);
    rewriter.replaceOpWithNewOp<NVVM::FenceProxyOp>(op, kind, spaceAttr);
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    // txl
    //int executingThreadId = getExecutingThreadId(op);

    auto id = getThreadId(rewriter, loc);
    auto pred = b.icmp_eq(id, b.i32_val(0)); // txl
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                            std::to_string(op.getCount()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InvalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InvalBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InvalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    // txl
    //int executingThreadId = getExecutingThreadId(op);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0)); // txl
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierExpectConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::BarrierExpectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierExpectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    // txl
    //int executingThreadId = getExecutingThreadId(op);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0)); // txl
    pred = b.and_(pred, adaptor.getPred());
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx =
        "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], " +
        std::to_string(op.getSize()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WaitBarrierOp> {
  const NVIDIA::TargetInfo *targetInfo;
  WaitBarrierOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto loc = op.getLoc();
    bool predicated =
        adaptor.getPred() && !matchPattern(op.getPred(), m_NonZero());
    std::string ptx;
    if (targetInfo->getComputeCapability() < 90) {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    } else {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    }
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create<>(ptx);
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 3> operands = {
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};
    if (predicated)
      operands.push_back(ptxBuilder.newOperand(adaptor.getPred(), "b"));

    waitLoop(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Add phase result as needed.
    std::stringstream ptxAsm;
    ptxAsm << "@$0 mbarrier.arrive.shared::cta.b64 _, [$1]";
    if (op.getCount() > 1) {
      ptxAsm << ", " << op.getCount();
    }
    ptxAsm << ";";

    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value id = getThreadId(rewriter, op.getLoc());
    Value pred = b.icmp_eq(id, b.i32_val(0));
    if (op.getPred())
      pred = b.and_(pred, adaptor.getPred());

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 2> operands = {
        ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(adaptor.getAlloc(), "r")};

    auto arriveOp = *ptxBuilder.create<>(ptxAsm.str());
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

// txl
struct MbarArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::MbarArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::MbarArriveOp>::ConvertOpToLLVMPattern;

  std::string getPtxAsm(tt::MbarArriveOp op, ttx::MBarriveType arriveType) const {
    Value ctaId = op.getRemoteCtaId();
    uint32_t txCount = op.getTxCount();
    std::string ptxAsm;
    switch (arriveType) {
    case ttx::MBarriveType::normal:
      ptxAsm = "@$1 mbarrier.arrive.shared.b64 _, [$0];";
      break;
    case ttx::MBarriveType::cp_async:
      ptxAsm = "@$1 cp.async.mbarrier.arrive.noinc.shared.b64 [$0];";
      break;
    case ttx::MBarriveType::expect_tx:
      assert(txCount > 0 && "txCount should be valid");
      ptxAsm = "@$1 mbarrier.arrive.expect_tx.shared.b64 _, [$0], " +
               std::to_string(txCount) + ";";
      break;
    case ttx::MBarriveType::remote:
      assert(ctaId && "ctaId should have a valid value");
      ptxAsm =
          " { .reg .b32 remAddr32;                                       \n"
          "  @$2 mapa.shared::cluster.u32  remAddr32, $0, $1;            \n"
          "  @$2 mbarrier.arrive.shared::cluster.b64  _, [remAddr32]; }  \n";
      break;
    default:
      llvm::errs() << "Unsupported mbarrier arrive type " << arriveType << "\n";
      llvm_unreachable("");
      break;
    }
    return ptxAsm;
  }

  LogicalResult
  matchAndRewrite(triton::MbarArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto mbarrier = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getOperands()[0],
        typeConverter->convertType(
            cast<mlir::triton::gpu::MemDescType>(op.getOperand(0).getType()).getElementType()
        ),
        rewriter);

    ::mlir::triton::PTXBuilder ptxBuilder;

    SmallVector<::mlir::triton::PTXBuilder::Operand*> operands;

    bool trackAsyncOp = op.getTrackAsyncOp();
    triton::txlgpu::MBarriveType type = triton::txlgpu::MBarriveType::normal;
    uint32_t txCount = op.getTxCount();
    auto remoteCtaId = adaptor.getRemoteCtaId();
    Value pred = adaptor.getPred();
    if (pred == nullptr) {
      pred = b.int_val(/*width*/ 1, 1);
    }

    if (trackAsyncOp) {
      type = triton::txlgpu::MBarriveType::cp_async;
    } else if (remoteCtaId) {
      assert(txCount == 0 &&
             "remote arrive of transaction mbarrier is not implemented yet");
      type = triton::txlgpu::MBarriveType::remote;
    } else if (txCount > 0) {
      type = triton::txlgpu::MBarriveType::expect_tx;
    }
    const std::string ptx = getPtxAsm(op, type);
    auto &mbarArriveOp = *ptxBuilder.create<>(ptx);

    switch (type) {
    case ttx::MBarriveType::normal:
    case ttx::MBarriveType::cp_async:
    case ttx::MBarriveType::expect_tx:
      operands.push_back(ptxBuilder.newOperand(mbarrier.getBase(), "r"));
      operands.push_back(ptxBuilder.newOperand(pred, "b"));
      break;
    case ttx::MBarriveType::remote:
      operands.push_back(ptxBuilder.newOperand(mbarrier.getBase(), "r"));
      operands.push_back(ptxBuilder.newOperand(remoteCtaId, "r"));
      operands.push_back(ptxBuilder.newOperand(pred, "b"));
      break;
    default:
      llvm::errs() << "Unsupported mbarrier arrive type " << type << "\n";
      llvm_unreachable("");
      break;
    }

    mbarArriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  // NOTE: txl
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<MbarArriveOpConversion>(typeConverter, benefit);

  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(typeConverter,
                                                                  benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
}
