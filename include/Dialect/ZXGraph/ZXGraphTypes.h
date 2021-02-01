#ifndef ZXGRAPH_ZXTYPES_H
#define ZXGRAPH_ZXTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace ZXGraph {

class NodeType : public Type::TypeBase<NodeType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace ZXGraph
} // namespace mlir

#endif // ZXGRAPH_ZXTYPES_H
