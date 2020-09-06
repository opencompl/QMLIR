//===- QuantumTypes.td - Quantum dialect Types -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

//===----------------------------------------------------------------------===//
// Qubit Type
//===----------------------------------------------------------------------===//
def Quantum_QubitType :
    Type<CPred<"$_self.isa<QubitType>()">, "Quantum Qubit Type">;


#endif // QUANTUM_TYPES