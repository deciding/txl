#include "txl/Dialect/TXL/IR/Dialect.h"
#include "txl/Dialect/TXL/IR/Dialect.cpp.inc"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::triton;

#define GET_ATTRDEF_CLASSES
#include "txl/Dialect/TXL/IR/TXLAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TritonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

void TXLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "txl/Dialect/TXL/IR/TXLAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "txl/Dialect/TXL/IR/Ops.cpp.inc"
      >();

}
