#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/TXLGPU/IR/Dialect.h"
#include "Dialect/TXLGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::txlgpu;

void mlir::triton::txlgpu::TXLGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TXLGPU/IR/TXLGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/TXLGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/TXLGPU/IR/Ops.cpp.inc"
