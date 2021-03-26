#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMOps.h"
#include "Dialect/QASM/QASMTypes.h"

using namespace mlir;
using namespace mlir::QASM;

namespace {
/// This class defines the interface for handling inlining with qssa
/// operations.
struct QASMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void QASMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/QASM/QASMOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/QASM/QASMOpsTypes.cpp.inc"
      >();
  addInterfaces<QASMInlinerInterface>();
}

Operation *QASMDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}
