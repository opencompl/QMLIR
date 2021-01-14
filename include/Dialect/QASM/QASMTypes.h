#ifndef QASM_QASMTYPES_H
#define QASM_QASMTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace QASM {

namespace detail {
struct QubitTypeStorage;
} // namespace detail

class QubitType
    : public Type::TypeBase<QubitType, Type, detail::QubitTypeStorage> {
public:
  using Base::Base;

  static QubitType get(MLIRContext *ctx, uint64_t size);

  uint64_t getSize() const;
};

} // namespace QASM
} // namespace mlir

#endif // QASM_QASMTYPES_H
