#ifndef DIALECT_QUANTUM_TYPEDETAIL_H_
#define DIALECT_QUANTUM_TYPEDETAIL_H_

#include "Dialect/Quantum/QuantumTypes.h"

namespace mlir {
namespace quantum {
namespace detail {

struct QubitTypeStorage : public TypeStorage {
  using KeyTy = int64_t;

  QubitTypeStorage(int64_t size) : size(size) { memRefShape.push_back(size); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(size); }

  static QubitTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<QubitTypeStorage>()) QubitTypeStorage(key);
  }

  // number of qubits in the array
  int64_t size;
  SmallVector<int64_t, 1> memRefShape;
};

} // namespace detail
} // namespace quantum
} // namespace mlir

#endif // DIALECT_QUANTUM_TYPEDETAIL_H_
