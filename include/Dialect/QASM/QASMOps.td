//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QASM_OPS
#define QASM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "QASMDialect.td"
include "QASMTypes.td"

def QASM_AllocateOp : QASM_Op<"allocate", []> {
  let summary = "Allocate an array of qubits.";
  let description = [{
    %q0 = qasm.allocate : !qasm.qubit<10>
  }];

  let results = (outs QASM_QubitType:$qout);

  let assemblyFormat = [{
    attr-dict `:` type($qout)
  }];

  let verifier = [{
    return success();
  }];
}

def QASM_ResetOp : QASM_Op<"reset", []> {
  let summary = "Reset a qubit array";
  let description = [{
    qasm.reset %q : !qasm.qubit<10>
  }];

  let arguments = (ins QASM_QubitType:$qinp);

  let assemblyFormat = [{
    $qinp attr-dict `:` type($qinp)
  }];

  let verifier = [{
    return success();
  }];
}

def QASM_MeasureOp : QASM_Op<"measure", []> {
  let summary = "Reset a qubit array";
  let description = [{
    qasm.reset %q : !qasm.qubit<10>
  }];

  let arguments = (ins QASM_QubitType:$qinp, MemRef);

  let assemblyFormat = [{
    $qinp attr-dict `:` type($qinp)
  }];

  let verifier = [{
    return success();
  }];
}

//======================
// Builtin Gates
//======================
def QASM_ControlledNotOp : QASM_Op<"CX", [SameTypeOperands]> {
  let summary = "Apply the C-NOT gate on two qubits";
  let description = [{
    ```mlir
    CX %0, %1 : !qasm.qubit<1>
    ```
  }];

  let arguments = (ins
    QASM_SingleQubitType:$qinp0,
    QASM_SingleQubitType:$qinp1);

  let assemblyFormat = [{
    $qinp0 `,` $qinp1 attr-dict `:` type($qinp0)
  }];

  let verifier = [{
    return success();
  }];
}

def QASM_SingleQubitRotationOp : QASM_Op<"U", []> {
  let summary = "Apply the C-NOT gate on two qubits";
  let description = [{
    ```mlir
    U(%theta : f32, %phi : f32, %lambda : f32) %q : !qasm.qubit<1>
    ```
    Note: parameters can be of any floating type
  }];

  let arguments = (ins
    QASM_QubitType:$qinp,
    AnyFloat:$theta,
    AnyFloat:$phi,
    AnyFloat:$lambda);

  let assemblyFormat = [{
    `(`
    $theta `:` type($theta) `,`
    $phi `:` type($phi) `,`
    $lambda `:` type($lambda)
    `)` $qinp attr-dict `:` type($qinp)
  }];

  let verifier = [{
    return success();
  }];
}

def QASM_GlobalPhaseGateOp : QASM_Op<"gphase", []> {
  let summary = "Apply a global phase.";
  let description = [{
    ```mlir
    U(%gamma : f32)
    ```
    Note: parameter can be of any floating type
  }];

  let arguments = (ins AnyFloat:$gamma);

  let assemblyFormat = [{
    `(` $gamma `:` type($gamma) `)` attr-dict
  }];

  let verifier = [{
    return success();
  }];
}

#endif // QASM_OPS
