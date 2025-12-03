#ifndef TXL_DIALECT_TXL_IR_DIALECT_H_
#define TXL_DIALECT_TXL_IR_DIALECT_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
//#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "txl/Dialect/TXL/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "txl/Dialect/TXL/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "txl/Dialect/TXL/IR/Ops.h.inc"

namespace mlir {
namespace triton { // not txl

struct SmemMemory : public SideEffects::Resource::Base<SmemMemory> {
  StringRef getName() final { return "<SmemMemory>"; }
};

} // namespace triton
} // namespace mlir

#endif // TXL_IR_DIALECT_H_
