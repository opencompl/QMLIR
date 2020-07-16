//===- QuantumOps.td - Quantum dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
def Quantum_QubitType :
    Type<CPred<"$_self.isa<QubitType>()">, "Quantum Qubit Type">;

def Quantum_GateType :
    Type<CPred<"$_self.isa<GateType>()">, "Quantum Gate Type">;

//===----------------------------------------------------------------------===//
// Allocate Op
//===----------------------------------------------------------------------===//

def Quantum_AllocateOp : Quantum_Op<"allocate", []> {
    let summary = "Allocate an array of qubits.";
    let description = [{
        This operation takes an unsigned integer argument `N`
        aNd returns a qubit array `qubit<N>`

        Example:

        ```mlir
        // allocate 10 qubits
        %qs = quantum.allocate : !quantum.qubit<10>
        ```
    }];

    let results = (outs Quantum_QubitType:$qreg);

    let assemblyFormat = [{
        attr-dict `:` type($qreg)
    }];
}

//===----------------------------------------------------------------------===//
// Concat Op
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Split Op
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Transform Op
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Measure Op
//===----------------------------------------------------------------------===//


#endif // QUANTUM_OPS
