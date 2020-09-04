//===- QuantumTypes.h - Quantum Types ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_QUANTUMTYPES_H
#define QUANTUM_QUANTUMTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace quantum {

namespace detail {
struct QubitTypeStorage;
} // namespace detail

// Qubit type: qubit<n>, n is an unsigned integer
class QubitType : public Type::TypeBase<QubitType, Type,
                                          detail::QubitTypeStorage> {
public:
  using Base::Base;

  static QubitType get(MLIRContext *ctx, int64_t size);

  // Return true iff the qubit has a fixed size
  bool hasStaticSize() const;

  // Return the size
  int64_t getSize() const;

  // Type equivalents for lowering to std
  ArrayRef<int64_t> getMemRefShape() const;
  Type getMemRefType() const;

  // For unknown-sized qubit arrays (`qubit<?>`)
  static constexpr int64_t kDynamicSize = ShapedType::kDynamicSize;
};

} // namespace quantum
} // namespace mlir

#endif // QUANTUM_QUANTUMTYPES_H
