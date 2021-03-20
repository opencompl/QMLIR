#ifndef QASM_QASMTYPES_H
#define QASM_QASMTYPES_H

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace QASM {

class QubitType : public Type::TypeBase<QubitType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace QASM
} // namespace mlir

#endif // QASM_QASMTYPES_H
