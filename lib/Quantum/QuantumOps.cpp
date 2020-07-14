//===- QuantumOps.cpp - Quantum dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quantum/QuantumOps.h"
#include "Quantum/QuantumDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace quantum {
#define GET_OP_CLASSES
#include "Quantum/QuantumOps.cpp.inc"
} // namespace quantum
} // namespace mlir
