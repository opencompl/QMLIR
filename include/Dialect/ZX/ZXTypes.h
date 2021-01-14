#ifndef ZX_ZXTYPES_H
#define ZX_ZXTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace ZX {

class WireType : public Type::TypeBase<WireType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace ZX
} // namespace mlir

#endif // ZX_ZXTYPES_H
