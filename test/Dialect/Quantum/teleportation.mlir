//===-teleportation.mlir - Quantum Teleporation Algorithm -----------------===//
// Basic teleportation algorithm.
// Subroutine @teleport takes in one source qubit in state |psi> with Alice
// and an entangled qubit pair shared by Alice and Bob.
// At the end, Bob's qubit has the sourcestate 
//===----------------------------------------------------------------------===//

// RUN: quantum-opt %s | quantum-opt 
// RUN: quantum-opt %s --inline --cse --memref-dataflow-opt

func private @std_to_bell(%qs: !qssa.qubit<2>) -> !qssa.qubit<2> {
  // H(qs[0])
  %q0, %q1 = qssa.split %qs : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %q2 = qssa.H %q0 : !qssa.qubit<1>

  // CNOT(qs[0], qs[1])
  %q3, %q4 = qssa.CNOT %q2, %q1
  %q5 = qssa.concat %q3, %q4 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  return %q5 : !qssa.qubit<2>
}

func private @bell_to_std(%qs : !qssa.qubit<2>) -> !qssa.qubit<2> {
  %q1:2 = qssa.split %qs : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)

  // CNOT(qs[0], qs[1])
  %q2:2 = qssa.CNOT %q1#0, %q1#1

  // H(qs[0])
  %q3_0 = qssa.H %q2#0 : !qssa.qubit<1>

  %q4 = qssa.concat %q3_0, %q2#1 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  return %q4 : !qssa.qubit<2>
}

func private @teleport(%psiA: !qssa.qubit<1>, %eb: !qssa.qubit<2>) -> (!qssa.qubit<1>) {
  %ebA, %psiB0 = qssa.split %eb : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)

  // Alice's qubits
  %qsA0 = qssa.concat %psiA, %ebA : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  // Measure in Bell basis
  %qsA1 = call @bell_to_std(%qsA0) : (!qssa.qubit<2>) -> !qssa.qubit<2>
  %resA, %qsA2 = qssa.measure %qsA1 : !qssa.qubit<2> -> tensor<2xi1>

  // Apply corrections

  // 1. Apply X correction, if memA[0] == 1
  %zero = constant 0 : index
  %corrX = tensor.extract %resA[%zero] : tensor<2xi1>

  %psiB1 = scf.if %corrX -> !qssa.qubit<1> {
    %temp = qssa.X %psiB0 : !qssa.qubit<1>
    scf.yield %temp : !qssa.qubit<1>
  } else {
    scf.yield %psiB0 : !qssa.qubit<1>
  }

  // 2. Apply Z correction, if memA[1] == 1
  %one = constant 1 : index
  %corrZ = tensor.extract %resA[%one] : tensor<2xi1>

  %psiB2 = scf.if %corrZ -> !qssa.qubit<1> {
    %temp = qssa.Z %psiB1 : !qssa.qubit<1>
    scf.yield %temp : !qssa.qubit<1>
  } else {
    scf.yield %psiB1 : !qssa.qubit<1>
  }

  return %psiB2 : !qssa.qubit<1>
}

func private @prepare_bell(%qa : !qssa.qubit<1>, %qb : !qssa.qubit<1>) -> !qssa.qubit<2> {
  %q0 = qssa.concat %qa, %qb : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  %q1 = call @std_to_bell(%q0) : (!qssa.qubit<2>) -> !qssa.qubit<2>
  return %q1 : !qssa.qubit<2>
}

func @main() {
  // Alice's qubits
  %psiA = qssa.alloc() : !qssa.qubit<1>
  %ebA = qssa.alloc() : !qssa.qubit<1>

  // Bob's qubits
  %ebB = qssa.alloc() : !qssa.qubit<1>

  // Entangle the qubits
  %eb = call @prepare_bell(%ebA, %ebB) : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  // Teleport |psi> from Alice to Bob
  %psiB = call @teleport(%psiA, %eb) : (!qssa.qubit<1>, !qssa.qubit<2>) -> !qssa.qubit<1>
  qssa.sink %psiB : !qssa.qubit<1>

  return
}
