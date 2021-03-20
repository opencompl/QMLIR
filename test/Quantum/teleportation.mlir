//===-teleportation.mlir - Quantum Teleporation Algorithm -----------------===//
// Basic teleportation algorithm.
// Subroutine @teleport takes in one source qubit in state |psi> with Alice
// and an entangled qubit pair shared by Alice and Bob.
// At the end, Bob's qubit has the sourcestate 
//===----------------------------------------------------------------------===//

// RUN: quantum-opt %s | quantum-opt 

func @std_to_bell(%qs: !qssa.qubit<2>) -> !qssa.qubit<2> {
  // H(qs[0])
  %q0, %q1 = qssa.split %qs : !qssa.qubit<2> -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %q2 = qssa.H %q0 : !qssa.qubit<1>

  // CNOT(qs[0], qs[1])
  %q3, %q4 = qssa.CNOT %q2, %q1
  %q5 = qssa.concat %q3, %q4 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  return %q5 : !qssa.qubit<2>
}

func @bell_to_std(%qs : !qssa.qubit<2>) -> !qssa.qubit<2> {
  %q1:2 = qssa.split %qs : !qssa.qubit<2> -> (!qssa.qubit<1>, !qssa.qubit<1>)

  // CNOT(qs[0], qs[1])
  %q2:2 = qssa.CNOT %q1#0, %q1#1

  // H(qs[0])
  %q3_0 = qssa.H %q2#0 : !qssa.qubit<1>

  %q4 = qssa.concat %q3_0, %q2#1 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  return %q4 : !qssa.qubit<2>
}

func @teleport(%psiA: !qssa.qubit<1>, %eb: !qssa.qubit<2>) -> (!qssa.qubit<1>) {
  %ebA, %psiB0 = qssa.split %eb : !qssa.qubit<2> -> (!qssa.qubit<1>, !qssa.qubit<1>)

  // Alice's qubits
  %qsA0 = qssa.concat %psiA, %ebA : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  // Measure in Bell basis
  %qsA1 = call @bell_to_std(%qsA0) : (!qssa.qubit<2>) -> !qssa.qubit<2>
  %resA = qssa.measure %qsA1 : !qssa.qubit<2> -> memref<2xi1>

  // Apply corrections

  // 1. Apply X correction, if resA[0] == 1
  %idx0 = constant 0 : index
  %corrX = load %resA[%idx0] : memref<2xi1>

  %psiB1 = scf.if %corrX -> !qssa.qubit<1> {
    %temp = qssa.X %psiB0 : !qssa.qubit<1>
    scf.yield %temp : !qssa.qubit<1>
  } else {
    scf.yield %psiB0 : !qssa.qubit<1>
  }

  // 2. Apply Z correction, if resA[1] == 1
  %idx1 = constant 1 : index
  %corrZ = load %resA[%idx1] : memref<2xi1>

  %psiB2 = scf.if %corrZ -> !qssa.qubit<1> {
    %temp = qssa.Z %psiB1 : !qssa.qubit<1>
    scf.yield %temp : !qssa.qubit<1>
  } else {
    scf.yield %psiB1 : !qssa.qubit<1>
  }

  return %psiB2 : !qssa.qubit<1>
}

func @prepare_bell(%qa : !qssa.qubit<1>, %qb : !qssa.qubit<1>) -> !qssa.qubit<2> {
  %q0 = qssa.concat %qa, %qb : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  %q1 = call @std_to_bell(%q0) : (!qssa.qubit<2>) -> !qssa.qubit<2>
  return %q1 : !qssa.qubit<2>
}

func @main() {
  // Alice's qubits
  %psiA = qssa.allocate() : !qssa.qubit<1>
  %ebA = qssa.allocate() : !qssa.qubit<1>

  // Bob's qubits
  %ebB = qssa.allocate() : !qssa.qubit<1>

  // Entangle the qubits
  %eb = call @prepare_bell(%ebA, %ebB) : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>

  // Teleport |psi> from Alice to Bob
  %psiB = call @teleport(%psiA, %eb) : (!qssa.qubit<1>, !qssa.qubit<2>) -> !qssa.qubit<1>

  return
}
