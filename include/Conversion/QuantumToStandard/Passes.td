//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD
#define CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD

include "mlir/Pass/PassBase.td"

def QuantumToStandardPass : Pass<"convert-qssa-to-std", "ModuleOp"> {
  let summary = "Convert quantum dialect to standard operations";
  let constructor = "mlir::createConvertQuantumToStandardPass()";
}

#endif // CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD
