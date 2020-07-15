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

namespace mlir {
namespace quantum {

namespace QuantumTypes {

enum Kinds {
  // use an experimental kind
  Qubit = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};

} // namespace QuantumTypes

namespace detail {
struct QubitTypeStorage;
} // namespace detail

// Qubit type: qubit<n>, n is an unsigned integer
class QubitType : public Type::TypeBase<QubitType, Type,
                                          detail::QubitTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == QuantumTypes::Qubit; }

  static QubitType get(MLIRContext *ctx, unsigned size);

  unsigned getSize() const;

};

} // namespace quantum
} // namespace mlir

#endif // QUANTUM_QUANTUMTYPES_H
