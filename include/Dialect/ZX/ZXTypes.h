#ifndef ZX_ZXTYPES_H
#define ZX_ZXTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/ZX/ZXOpsTypes.h.inc"

#endif // ZX_ZXTYPES_H
