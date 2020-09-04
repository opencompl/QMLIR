//===-teleportation.mlir - Quantum Teleporation Algorithm -----------------===//
// Basic teleportation algorithm.
// Subroutine @teleport takes in one source qubit in state |psi> with Alice
// and an entangled qubit pair shared by Alice and Bob.
// At the end, Bob's qubit has the sourcestate 
//===----------------------------------------------------------------------===//

// RUN: quantum-opt %s | quantum-opt 

//#gateH = {
//  name = "H",
//  size = 1,
//  matrix = dense<
//    [[0.7071067811865476,  0.7071067811865476],
//     [0.7071067811865476, -0.7071067811865476]]> : tensor<2x2xf64>
//}
//
//#gateX = {
//  name = "X",
//  size = 1,
//  matrix = dense<
//    [[0.0, 1.0],
//     [1.0, 0.0]]> : tensor<2x2xf64>
//}
//
//#gateZ = {
//  name = "Z",
//  size = 1,
//  matrix = dense<
//    [[1.0,  0.0],
//     [0.0, -1.0]]> : tensor<2x2xf64>
//}
//
//#gateCNOT = {
//  name = "CNOT",
//  size = 2,
//  matrix = sparse<
//    [[0, 0], [1, 1], [2, 3], [3, 2]],
//    [ 1.0,    1.0,    1.0,    1.0  ]> : tensor<4x4xf64>
//}
//
//func @std_to_bell(%qs: !quantum.qubit<2>) -> !quantum.qubit<2> {
//  // H(qs[0])
//  %q0, %q1 = quantum.split %qs : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
//  %q2 = quantum.transform #gateH(%q0) : !quantum.qubit<1>
//
//  // CNOT(qs[0], qs[1])
//  %q3 = quantum.concat %q2, %q1 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
//  %q4 = quantum.transform #gateCNOT(%q3) : !quantum.qubit<2>
//
//  return %q4 : !quantum.qubit<2>
//}
//
//func @bell_to_std(%qs : !quantum.qubit<2>) -> !quantum.qubit<2> {
//  // CNOT(qs[0], qs[1])
//  %q0 = quantum.transform #gateCNOT(%qs) : !quantum.qubit<2>
//
//  // H(qs[0])
//  %q1, %q2 = quantum.split %q0 : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
//  %q3 = quantum.transform #gateH(%q1) : !quantum.qubit<1>
//
//  %q4 = quantum.concat %q3, %q2 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
//  return %q4 : !quantum.qubit<2>
//}
//
//func @teleport(%psiA: !quantum.qubit<1>, %eb: !quantum.qubit<2>) -> (!quantum.qubit<1>) {
//  %ebA, %psiB0 = quantum.split %eb : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
//
//  // Alice's qubits
//  %qsA0 = quantum.concat %psiA, %ebA : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
//
//  // Measure in Bell basis
//  %qsA1 = call @bell_to_std(%qsA0) : (!quantum.qubit<2>) -> !quantum.qubit<2>
//  %resA = quantum.measure %qsA1 : !quantum.qubit<2> -> memref<2xi1>
//
//  // Apply corrections
//
//  // 1. Apply X correction, if resA[0] == 1
//  %idx0 = constant 0 : index
//  %corrX = load %resA[%idx0] : memref<2xi1>
//
//  %psiB1 = scf.if %corrX -> !quantum.qubit<1> {
//    %temp = quantum.transform #gateX(%psiB0) : !quantum.qubit<1>
//    scf.yield %temp : !quantum.qubit<1>
//  } else {
//    scf.yield %psiB0 : !quantum.qubit<1>
//  }
//
//  // 2. Apply Z correction, if resA[1] == 1
//  %idx1 = constant 1 : index
//  %corrZ = load %resA[%idx1] : memref<2xi1>
//
//  %psiB2 = scf.if %corrZ -> !quantum.qubit<1> {
//    %temp = quantum.transform #gateZ(%psiB1) : !quantum.qubit<1>
//    scf.yield %temp : !quantum.qubit<1>
//  } else {
//    scf.yield %psiB1 : !quantum.qubit<1>
//  }
//
//  return %psiB2 : !quantum.qubit<1>
//}
//
//func @prepare_bell(%qa : !quantum.qubit<1>, %qb : !quantum.qubit<1>) -> !quantum.qubit<2> {
//  %q0 = quantum.concat %qa, %qb : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
//  %q1 = call @std_to_bell(%q0) : (!quantum.qubit<2>) -> !quantum.qubit<2>
//  return %q1 : !quantum.qubit<2>
//}
//
//func @main() {
//  // Alice's qubits
//  %psiA = quantum.allocate() : !quantum.qubit<1>
//  %ebA = quantum.allocate() : !quantum.qubit<1>
//
//  // Bob's qubits
//  %ebB = quantum.allocate() : !quantum.qubit<1>
//
//  // Entangle the qubits
//  %eb = call @prepare_bell(%ebA, %ebB) : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
//
//  // Teleport |psi> from Alice to Bob
//  %psiB = call @teleport(%psiA, %eb) : (!quantum.qubit<1>, !quantum.qubit<2>) -> !quantum.qubit<1>
//
//  return
//}
