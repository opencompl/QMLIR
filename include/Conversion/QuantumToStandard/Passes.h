//===- Passes.h - Quantum dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUANTUMTOSTANDARD_PASSES_H
#define CONVERSION_QUANTUMTOSTANDARD_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertQuantumToStandardPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/QuantumToStandard/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QUANTUMTOSTANDARD_PASSES_H
