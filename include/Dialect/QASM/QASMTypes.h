#ifndef QASM_QASMTYPES_H
#define QASM_QASMTYPES_H

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/QASM/QASMOpsTypes.h.inc"

#endif // QASM_QASMTYPES_H
