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
  using KeyTy = unsigned;

  QubitTypeStorage(unsigned size)
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
  unsigned size;
};

QubitType QubitType::get(MLIRContext *ctx, unsigned size) {
  return Base::get(ctx, QuantumTypes::Qubit, size);
}

unsigned QubitType::getSize() const {
  return getImpl()->size;
}
