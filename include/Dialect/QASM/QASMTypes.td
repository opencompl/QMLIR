//===- QuantumTypes.td - Quantum dialect Types -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QASM_TYPES
#define QASM_TYPES

include "mlir/IR/OpBase.td"

// !qasm.qubit
def QASM_QubitType : QASM_Type<"Qubit"> {
  let mnemonic = "qubit";
}

#endif // QASM_TYPES
