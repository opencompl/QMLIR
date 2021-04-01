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
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "QASMDialect.td"
include "QASMTypes.td"

def QASM_AllocateOp : QASM_Op<"allocate", []> {
  let summary = "Allocate a qubit";
  let description = [{
    %q = qasm.allocate
  }];

  let results = (outs QASM_QubitType:$qout);
  let assemblyFormat = [{ attr-dict }];
  let verifier = ?;
}

def QASM_ResetOp : QASM_Op<"reset", []> {
  let summary = "Reset a qubit";
  let description = [{
    qasm.reset %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);
  let assemblyFormat = [{ $qinp attr-dict }];
  let verifier = ?;
}

def QASM_MeasureOp : QASM_Op<"measure", []> {
  let summary = "Measure a qubit array";
  let description = [{
    %res = qasm.measure %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);
  let results = (outs I1:$res);
  let assemblyFormat = [{ $qinp attr-dict }];
  let verifier = ?;
}

def QASM_BarrierOp : QASM_Op<"barrier", []> {
  let summary = "Add a barrier for a qubit";
  let description = [{
    Prevents reordering operations containing the qubit across the barrier
    qasm.barrier %q
  }];

  let arguments = (ins QASM_QubitType:$qinp);
  let assemblyFormat = [{ $qinp attr-dict }];
  let verifier = ?;
}

def QASM_PIOp : QASM_Op<"pi", [NoSideEffect]> {
  let summary = "Return the value of pi";
  let description = [{
    %pi = qasm.pi : f32
    // approximately equivalent to
    %pi = constant 3.142 : f32
  }];

  let results = (outs AnyFloat:$qout);
  let assemblyFormat = [{ attr-dict `:` type($qout) }];
  let verifier = ?;
  let hasFolder = 1;
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

  let arguments = (ins QASM_QubitType:$qinp0, QASM_QubitType:$qinp1);
  let assemblyFormat = [{ $qinp0 `,` $qinp1 attr-dict }];
  let verifier = ?;
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

  let verifier = ?;
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
  let verifier = ?;
}

// Custom Gate Call
def QASM_GateCall : QASM_Op<"gate", []> {
  let summary = "Add opaque call to a gate";
  let description = [{
    ```mlir
    qasm.gate @name(%alpha, %q) : (f32, !qasm.qubit) -> ()
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$gate_name,
    Variadic<AnyTypeOf<[AnyFloat, QASM_QubitType]>>:$gate_args);
  let assemblyFormat = [{
    $gate_name `(` $gate_args `)` attr-dict `:` functional-type($gate_args, results)
  }];
  let verifier = ?;
}

//================== Basic Control Flow ===================================//

def QASM_CregType : MemRefRankOf<[I1], [1]>;
def QASM_IfOp : QASM_Op<"if", [SingleBlockImplicitTerminator<"EndIfOp">,
                               RecursiveSideEffects,
                               NoRegionArguments]> {
  let summary = "QASM If Operation";
  let description = [{
    ```mlir
    // N :: Integer constant (<= 64)
    // %creg : memref<Nxi1>
    // V :: I64Attr
    qasm.if %creg V {
      qasm.CX %a0, %b0
      qasm.CX %a1, %b1
      qasm.CX %a2, %b2
      qasm.CX %a3, %b3
    }
  }];
  let arguments = (ins QASM_CregType:$creg, I64Attr:$value);
  let regions = (region SizedRegion<1>:$ifBlock);
  let assemblyFormat = [{ $creg `=` $value attr-dict `:` type($creg) $ifBlock }];
}

def QASM_EndIfOp : QASM_Op<"end_if", [Terminator, ReturnLike,
                                      ParentOneOf<["IfOp"]>, NoSideEffect]> {
  let summary = "QASM if block terminator";
  let description = [{
    ```mlir
    qasm.if %creg == 0 {
      qasm.CX %0, %1
      qasm.CX %2, %3
      qasm.end_if
    }
    ```
    Will be implicitly added if not specified.
  }];
  let verifier = ?;
  let assemblyFormat = [{ attr-dict }];
}

#endif // QASM_OPS
