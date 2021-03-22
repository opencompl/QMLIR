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
#include "TypeDetail.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Qubit Type
//===----------------------------------------------------------------------===//

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
