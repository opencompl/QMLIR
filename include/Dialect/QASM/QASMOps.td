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
  let summary = "Allocate a qubit";
  let description = [{
    %q = qasm.allocate
  }];

  let results = (outs QASM_QubitType:$qout);

  let assemblyFormat = [{ attr-dict }];

  let verifier = [{
    return success();
  }];
}

def QASM_ResetOp : QASM_Op<"reset", []> {
  let summary = "Reset a qubit";
  let description = [{
    qasm.reset %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);

  let assemblyFormat = [{ $qinp attr-dict }];

  let verifier = [{
    return success();
  }];
}

def QASM_MeasureOp : QASM_Op<"measure", []> {
  let summary = "Measure a qubit array";
  let description = [{
    %res = qasm.measure %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);
  let results = (outs I1:$res);

  let assemblyFormat = [{ $qinp attr-dict }];

  let verifier = [{
    return success();
  }];
}

def QASM_BarrierOp : QASM_Op<"barrier", []> {
  let summary = "Add a barrier for a qubit";
  let description = [{
    Prevents reordering operations containing the qubit across the barrier
    qasm.barrier %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);

  let assemblyFormat = [{ $qinp attr-dict }];

  let verifier = [{
    return success();
  }];
}

def QASM_PIOp : QASM_Op<"pi", []> {
  let summary = "Return the value of pi";
  let description = [{
    %pi = qasm.pi : f32
    // approximately equivalent to
    %pi = constant 3.142 : f32
  }];

  let results = (outs AnyFloat:$qout);

  let assemblyFormat = [{ attr-dict `:` type($qout) }];

  let verifier = [{
    return success();
  }];
}

//======================
// Builtin Gates
//======================
def QASM_ControlledNotOp : QASM_Op<"CX", []> {
  let summary = "Apply the C-NOT gate on two qubits";
  let description = [{
    ```mlir
    qasm.CX %0, %1
    ```
  }];

  let arguments = (ins
    QASM_QubitType:$qinp0,
    QASM_QubitType:$qinp1);

  let assemblyFormat = [{ $qinp0 `,` $qinp1 attr-dict }];

  let verifier = [{
    return success();
  }];
}

def QASM_SingleQubitRotationOp : QASM_Op<"U", []> {
  let summary = "Apply a universal single qubit rotation";
  let description = [{
    ```mlir
    qasm.U(%theta : f32, %phi : f32, %lambda : f32) %q
    ```
    Note: parameters can be of any floating type.
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
    `)` $qinp attr-dict
  }];

  let verifier = [{
    return success();
  }];
}

def QASM_GlobalPhaseGateOp : QASM_Op<"gphase", []> {
  let summary = "Apply a global phase.";
  let description = [{
    ```mlir
    qasm.gphase(%gamma : f32)
    ```
    Note: parameter can be of any floating type
  }];

  let arguments = (ins AnyFloat:$gamma);

  let assemblyFormat = [{ `(` $gamma `:` type($gamma) `)` attr-dict }];

  let verifier = [{
    return success();
  }];
}

#endif // QASM_OPS
