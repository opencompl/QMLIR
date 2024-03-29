// RUN: quantum-opt --convert-qssa-to-qir %s | FileCheck %s

module {
  // CHECK-LABEL: @allocate_static
  func @allocate_static() {
    // CHECK: %[[SIZE:.*]] = constant 5 : index
    // CHECK-NEXT: %[[QUBITS:.*]] = call @__mlir_qssa_simulator__acquire_qubits(%[[SIZE]]) : (index) -> memref<?xi64>
    // CHECK-NEXT: %{{.*}} = memref_cast %[[QUBITS]] : memref<?xi64> to memref<5xi64>
    %q = qssa.allocate() : !qssa.qubit<5>
    return
  }

  // CHECK-LABEL: @allocate_dynamic
  func @allocate_dynamic() {
    // CHECK: %[[SIZE:.*]] = constant 4 : index
    %n = constant 4 : index
    // CHECK: %[[QUBITS:.*]] = call @__mlir_qssa_simulator__acquire_qubits(%[[SIZE]]) : (index) -> memref<?xi64>
    %q = qssa.allocate(%n) : !qssa.qubit<?>
    return
  }
}
