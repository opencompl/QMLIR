#include "Dialect/QASM/QASMOps.h"

using namespace mlir;
using namespace mlir::QASM;

static LogicalResult verify(IfOp op) {
  uint64_t numQubits = op.creg().getType().cast<MemRefType>().getShape()[0];
  if (numQubits > 64)
    return op.emitOpError("unsupported: size of register too large (> 64)");
  if ((op.value() >> numQubits) != 0)
    return op.emitOpError("result too large, must be in range [0, 2^qubits)");
  return success();
}

#define GET_OP_CLASSES
#include "Dialect/QASM/QASMOps.cpp.inc"

OpFoldResult PIOp::fold(ArrayRef<Attribute> operands) {
  return FloatAttr::get(getType(), M_PI);
}