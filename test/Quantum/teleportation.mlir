//===-teleportation.mlir - Quantum Teleporation Algorithm -----------------===//
// Basic teleportation algorithm.
// Subroutine @teleport takes in one source qubit in state |psi> with Alice
// and an entangled qubit pair shared by Alice and Bob.
// At the end, Bob's qubit has the sourcestate 
//===----------------------------------------------------------------------===//

// RUN: quantum-opt %s | quantum-opt 

module {

  func @teleport(%psiA: !quantum.qubit<1>, %eb: !quantum.qubit<2>) 
                -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) {
    %H = quantum.gate_prim {name = "H"}: !quantum.gate<1>
    %X = quantum.gate_prim {name = "X"}: !quantum.gate<1>
    %Z = quantum.gate_prim {name = "Z"}: !quantum.gate<1>
    %CNOT = quantum.gate_prim {name = "CNOT"}: !quantum.gate<2>
    
    %ebA, %ebB = quantum.split (%eb: !quantum.qubit<2>) : (!quantum.qubit<1>, !quantum.qubit<1>)
    
    // Alice's qubits
    %qsA = quantum.concat (%psiA: !quantum.qubit<1>), (%ebA: !quantum.qubit<1>) : !quantum.qubit<2>

    // Measure in Bell basis
    %qsA_1 = quantum.transform (%qsA : !quantum.qubit<2>), (%CNOT : !quantum.gate<2>) : !quantum.qubit<2>
    %qA1_1, %qA2_1 = quantum.split (%qsA_1 : !quantum.qubit<2>) : (!quantum.qubit<1>, !quantum.qubit<1>)
    %qA1_2 = quantum.transform (%qA1_1 : !quantum.qubit<1>), (%H : !quantum.gate<1>) : !quantum.qubit<1>
    %res1 = quantum.measure (%qA1_2 : !quantum.qubit<1>) : i64
    %res2 = quantum.measure (%qA2_1 : !quantum.qubit<1>) : i64

    // Apply corrections
    %one = constant 1 : i64

    // Apply X correction, if res1 == 1
    %use_X = cmpi "eq", %res1, %one : i64
    cond_br %use_X, ^ifX, ^elseX

  ^ifX:
    %psiB_temp1 = quantum.transform (%ebB: !quantum.qubit<1>), (%X : !quantum.gate<1>) : !quantum.qubit<1>
    br ^doneX(%psiB_temp1: !quantum.qubit<1>)

  ^elseX:
    br ^doneX(%ebB: !quantum.qubit<1>)
    
  ^doneX(%psiB_1 : !quantum.qubit<1>):

    // Apply Z correction, if res2 == 1
    %use_Z = cmpi "eq", %res2, %one : i64
    cond_br %use_Z, ^ifZ, ^elseZ

  ^ifZ:
    %psiB_temp2 = quantum.transform (%psiB_1: !quantum.qubit<1>), (%Z : !quantum.gate<1>) : !quantum.qubit<1>
    br ^doneZ(%psiB_temp2: !quantum.qubit<1>)

  ^elseZ:
    br ^doneZ(%psiB_1: !quantum.qubit<1>)
    
  ^doneZ(%psiB_2 : !quantum.qubit<1>):

    return %qA1_1, %qA1_2, %psiB_2 : !quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>
  }

  func @prepare_bell(%qa: !quantum.qubit<1>, %qb: !quantum.qubit<1>) -> !quantum.qubit<2> {
    %H = quantum.gate_prim {name = "H"}: !quantum.gate<1>
    %CNOT = quantum.gate_prim {name = "CNOT"}: !quantum.gate<2>
    
    // H(qa)
    %qa_1 = quantum.transform (%qa: !quantum.qubit<1>), (%H: !quantum.gate<1>) : !quantum.qubit<1>
    %eb = quantum.concat (%qa_1 : !quantum.qubit<1>), (%qb : !quantum.qubit<1>) : !quantum.qubit<2>

    // CNOT(qa, qb)
    %eb_1 = quantum.transform (%eb : !quantum.qubit<2>), (%CNOT : !quantum.gate<2>) : !quantum.qubit<2>

    return %eb_1 : !quantum.qubit<2>
  }

  func @main() {
    // Alice's qubits
    %psiA = quantum.allocate : !quantum.qubit<1>
    %ebA = quantum.allocate : !quantum.qubit<1>

    // Bob's qubits
    %ebB = quantum.allocate : !quantum.qubit<1>
    
    // Entangle the qubits
    %eb = call @prepare_bell(%ebA, %ebB) : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
    
    // Teleport |psi> from Alice to Bob
    %ebA_1, %ebA_2, %psiB = call @teleport(%psiA, %eb)
                              : (!quantum.qubit<1>, !quantum.qubit<2>) 
                                  -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)

    return
  }
}
