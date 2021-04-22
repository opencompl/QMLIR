//===- Passes.h - Quantum dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUANTUMTOLLVM_PASSES_H
#define CONVERSION_QUANTUMTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertQuantumToLLVMPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/QuantumToLLVM/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QUANTUMTOLLVM_PASSES_H
