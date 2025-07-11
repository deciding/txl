#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "txl/Dialect/TXL/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {

void SmemAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "txl/Dialect/TXL/IR/Ops.cpp.inc"
