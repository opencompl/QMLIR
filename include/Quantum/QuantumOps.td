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
        This operation takes no arguments,
        and returns a qubit array `qubit<N>`.

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

def Quantum_ConcatOp : Quantum_Op<"concat", []> {
    let summary = "Concatenate two arrays of qubits.";
    let description = [{
        This operation takes two qubit arrays of types `qubit<N>` and 
        `qubit<M>`, and returns the concatenated array of type `qubit<N+M>`

        Example:

        ```mlir
        %q1 = quantum.allocate : !quantum.qubit<4>
        %q2 = quantum.allocate : !quantum.qubit<5>
        %q3 = quantum.concat %q1, %q2 : !quantum.qubit<9>
        ```
    }];

    let arguments = (ins 
        Quantum_QubitType:$qreg0, 
        Quantum_QubitType:$qreg1);

    let results = (outs Quantum_QubitType:$qreg);

    let assemblyFormat = [{
        `(` $qreg0 `:` type($qreg0) `)` `,` `(` $qreg1 `:` type($qreg1) `)` attr-dict `:` type($qreg)
    }];

    let verifier = [{
        if (qreg0().getType().cast<QubitType>().getSize()
            + qreg1().getType().cast<QubitType>().getSize() 
            != qreg().getType().cast<QubitType>().getSize())
            return emitOpError("sizes don't match!");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Split Op
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Transform Op
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Measure Op
//===----------------------------------------------------------------------===//

def Quantum_MeasureOp : Quantum_Op<"measure", []> {
    let summary = "Measure an array of qubits.";
    let description = [{
        This operation takes an array of qubits of type `qubit<N>`
        and returns the measurement outcome in the standard basis
        as a little-endian unsigned integer.

        Example:

        ```mlir
        %qs = quantum.allocate : !quantum.qubit<10>
        %res = quantum.measure (%qs : qubit<10>) : u64
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qreg);
    let results = (outs I64:$res);

    let assemblyFormat = [{
        `(` $qreg `:` type($qreg) `)` attr-dict `:` type($res)
    }];
}


#endif // QUANTUM_OPS
