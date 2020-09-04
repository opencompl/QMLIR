// RUN: quantum-opt --convert-quantum-to-std %s | quantum-opt | FileCheck %s

module {
  // CHECK-LABEL: func @cast_op_lowering(
  // CHECK-SAME: %[[ARG:.*]]: memref<5xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @cast_op_lowering(%0 : !quantum.qubit<5>) -> !quantum.qubit<?> {
    // CHECK: %[[RES:.*]] = memref_cast %[[ARG]] : memref<5xi64> to memref<?xi64>
    %1 = quantum.cast %0 : !quantum.qubit<5> to !quantum.qubit<?>
    // CHECK: return %[[RES]] : memref<?xi64>
    return %1 : !quantum.qubit<?>
  }


  // CHECK-LABEL: func @dim_op_lowering(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> (memref<?xi64>, index) {
  func @dim_op_lowering(%q : !quantum.qubit<?>) -> (!quantum.qubit<?>, index) {
    // CHECK: %[[RANK:.*]] = constant 0 : index
    // CHECK-NEXT: %[[SIZE:.*]] = dim %[[ARG]], %[[RANK]] : memref<?xi64>
    %q1, %n = quantum.dim %q : !quantum.qubit<?>
    // CHECK: return %[[ARG]], %[[SIZE]] : memref<?xi64>, index
    return %q1, %n : !quantum.qubit<?>, index
  }


  // CHECK-LABEL: func @concat_op_lowering(
  // CHECK-SAME: %[[ARG0:.*]]: memref<?xi64>, %[[ARG1:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @concat_op_lowering(%q0 : !quantum.qubit<?>, %q1 : !quantum.qubit<?>) -> !quantum.qubit<?> {
    // CHECK: %[[RES:.*]] = call @__mlir_quantum_simulator__concat_qubits(
    // CHECK-SAME: %[[ARG0]], %[[ARG1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    %q2 = quantum.concat %q0, %q1 : (!quantum.qubit<?>, !quantum.qubit<?>) -> !quantum.qubit<?>
    // CHECK: return %[[RES]] : memref<?xi64>
    return %q2 : !quantum.qubit<?>
  }

  // CHECK-LABEL: func @concat_op_lowering_static(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<3xi64>
  // CHECK-SAME: ) -> memref<5xi64> {
  func @concat_op_lowering_static(%q0 : !quantum.qubit<2>, %q1 : !quantum.qubit<3>) -> !quantum.qubit<5> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[INP1:.*]] = memref_cast %[[ARG1]] : memref<3xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_quantum_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[INP1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi64> to memref<5xi64>
    %q2 = quantum.concat %q0, %q1 : (!quantum.qubit<2>, !quantum.qubit<3>) -> !quantum.qubit<5>

    // CHECK: return %[[RES]] : memref<5xi64>
    return %q2 : !quantum.qubit<5>
  }

  // concat (static, static) -> dynamic
  // CHECK-LABEL: func @concat_op_lowering_mixed_1(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<3xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @concat_op_lowering_mixed_1(%q0 : !quantum.qubit<2>, %q1 : !quantum.qubit<3>) -> !quantum.qubit<?> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[INP1:.*]] = memref_cast %[[ARG1]] : memref<3xi64> to memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = call @__mlir_quantum_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[INP1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    %q2 = quantum.concat %q0, %q1 : (!quantum.qubit<2>, !quantum.qubit<3>) -> !quantum.qubit<?>

    // CHECK: return %[[RES]] : memref<?xi64>
    return %q2 : !quantum.qubit<?>
  }

  // concat (static, dynamic) -> static
  // CHECK-LABEL: func @concat_op_lowering_mixed_2(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> memref<5xi64> {
  func @concat_op_lowering_mixed_2(%q0 : !quantum.qubit<2>, %q1 : !quantum.qubit<?>) -> !quantum.qubit<5> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_quantum_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[ARG1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi64> to memref<5xi64>
    %q2 = quantum.concat %q0, %q1 : (!quantum.qubit<2>, !quantum.qubit<?>) -> !quantum.qubit<5>

    // CHECK: return %[[RES]] : memref<5xi64>
    return %q2 : !quantum.qubit<5>
  }


  // CHECK-LABEL: func @measure_op_lowering_dynamic(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> i1 {
  func @measure_op_lowering_dynamic(%q0 : !quantum.qubit<?>) -> i1 {
    // CHECK: %[[RES:.*]] = call @__mlir_quantum_simulator__measure_qubits(
    // CHECK-SAME: %[[ARG]]) : (memref<?xi64>) -> memref<?xi1>
    %res = quantum.measure %q0 : !quantum.qubit<?> -> memref<?xi1>

    %0 = constant 0 : index
    // CHECK: %{{.*}} = load %[[RES]][%{{.*}}] : memref<?xi1>
    %v = load %res[%0] : memref<?xi1>
    return %v : i1
  }

  // CHECK-LABEL: func @measure_op_lowering_static(
  // CHECK-SAME: %[[ARG:.*]]: memref<5xi64>
  // CHECK-SAME: ) -> i1 {
  func @measure_op_lowering_static(%q0 : !quantum.qubit<5>) -> i1 {
    // CHECK: %[[INP:.*]] = memref_cast %[[ARG]] : memref<5xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_quantum_simulator__measure_qubits(
    // CHECK-SAME: %[[INP]]) : (memref<?xi64>) -> memref<?xi1>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi1> to memref<5xi1>
    %res = quantum.measure %q0 : !quantum.qubit<5> -> memref<5xi1>

    %0 = constant 0 : index
    // CHECK: %{{.*}} = load %[[RES]][%{{.*}}] : memref<5xi1>
    %v = load %res[%0] : memref<5xi1>
    return %v : i1
  }
}
