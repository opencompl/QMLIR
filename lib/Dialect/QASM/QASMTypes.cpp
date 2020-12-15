#include "Dialect/QASM/QASMTypes.h"

using namespace mlir;
using namespace mlir::QASM;

struct QASM::detail::QubitTypeStorage : public TypeStorage {
  QubitTypeStorage(uint64_t size) : size(size) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = uint64_t;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const { return key == size; }

  /// Define a construction method for creating a new instance of this storage.
  static QubitTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<QubitTypeStorage>()) QubitTypeStorage(key);
  }

  /// The parametric data held by the storage class.
  uint64_t size;
};

QubitType QubitType::get(MLIRContext *ctx, uint64_t size) {
  return Base::get(ctx, size);
}
uint64_t QubitType::getSize() const { return getImpl()->size; }
