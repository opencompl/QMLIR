//===- QuantumTypes.cpp - Quantum Types  ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/Quantum/QuantumTypes.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Qubit Type
//===----------------------------------------------------------------------===//

struct quantum::detail::QubitTypeStorage : public TypeStorage {
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

QubitType QubitType::get(MLIRContext *ctx, int64_t size) {
  return Base::get(ctx, size);
}

bool QubitType::hasStaticSize() const {
  return getImpl()->size != kDynamicSize;
}

int64_t QubitType::getSize() const { return getImpl()->size; }

ArrayRef<int64_t> QubitType::getMemRefShape() const {
  return getImpl()->memRefShape;
}

Type QubitType::getMemRefType() const {
  return IntegerType::get(getContext(), 64);
}
