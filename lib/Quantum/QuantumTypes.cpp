//===- QuantumTypes.cpp - Quantum Types  ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quantum/QuantumTypes.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Qubit Type
//===----------------------------------------------------------------------===//

struct quantum::detail::QubitTypeStorage : public TypeStorage {
  using KeyTy = uint64_t;

  QubitTypeStorage(uint64_t size)
      : size(size) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(size);
  }

  static QubitTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<QubitTypeStorage>())
        QubitTypeStorage(key);
  }

  // number of qubits in the array
  uint64_t size;
};

QubitType QubitType::get(MLIRContext *ctx, uint64_t size) {
  return Base::get(ctx, size);
}

bool QubitType::hasStaticSize() const {
  return getImpl()->size != kDynamicSize;
}

uint64_t QubitType::getSize() const {
  return getImpl()->size;
}
