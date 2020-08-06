//===- QuantumOps.h - Quantum dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_QUANTUMOPS_H
#define QUANTUM_QUANTUMOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

#include "Quantum/QuantumTypes.h"

namespace mlir {
namespace quantum {

#define GET_OP_CLASSES
#include "Quantum/QuantumOps.h.inc"

} // namespace quantum
} // namespace mlir

#endif // QUANTUM_QUANTUMOPS_H
