// RUN: quantum-opt --convert-quantum-to-std %s | quantum-opt | FileCheck %s

module {
  // CHECK-LABEL: func @cast_op_lowering(
  // CHECK-SAME: %[[ARG:.*]]: memref<5xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @cast_op_lowering(%0 : !qssa.qubit<5>) -> !qssa.qubit<?> {
    // CHECK: %[[RES:.*]] = memref_cast %[[ARG]] : memref<5xi64> to memref<?xi64>
    %1 = qssa.cast %0 : !qssa.qubit<5> to !qssa.qubit<?>
    // CHECK: return %[[RES]] : memref<?xi64>
    return %1 : !qssa.qubit<?>
  }


  // CHECK-LABEL: func @dim_op_lowering(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> (memref<?xi64>, index) {
  func @dim_op_lowering(%q : !qssa.qubit<?>) -> (!qssa.qubit<?>, index) {
    // CHECK: %[[RANK:.*]] = constant 0 : index
    // CHECK-NEXT: %[[SIZE:.*]] = dim %[[ARG]], %[[RANK]] : memref<?xi64>
    %q1, %n = qssa.dim %q : !qssa.qubit<?>
    // CHECK: return %[[ARG]], %[[SIZE]] : memref<?xi64>, index
    return %q1, %n : !qssa.qubit<?>, index
  }


  // CHECK-LABEL: func @concat_op_lowering(
  // CHECK-SAME: %[[ARG0:.*]]: memref<?xi64>, %[[ARG1:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @concat_op_lowering(%q0 : !qssa.qubit<?>, %q1 : !qssa.qubit<?>) -> !qssa.qubit<?> {
    // CHECK: %[[RES:.*]] = call @__mlir_qssa_simulator__concat_qubits(
    // CHECK-SAME: %[[ARG0]], %[[ARG1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    %q2 = qssa.concat %q0, %q1 : (!qssa.qubit<?>, !qssa.qubit<?>) -> !qssa.qubit<?>
    // CHECK: return %[[RES]] : memref<?xi64>
    return %q2 : !qssa.qubit<?>
  }

  // CHECK-LABEL: func @concat_op_lowering_static(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<3xi64>
  // CHECK-SAME: ) -> memref<5xi64> {
  func @concat_op_lowering_static(%q0 : !qssa.qubit<2>, %q1 : !qssa.qubit<3>) -> !qssa.qubit<5> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[INP1:.*]] = memref_cast %[[ARG1]] : memref<3xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_qssa_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[INP1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi64> to memref<5xi64>
    %q2 = qssa.concat %q0, %q1 : (!qssa.qubit<2>, !qssa.qubit<3>) -> !qssa.qubit<5>

    // CHECK: return %[[RES]] : memref<5xi64>
    return %q2 : !qssa.qubit<5>
  }

  // concat (static, static) -> dynamic
  // CHECK-LABEL: func @concat_op_lowering_mixed_1(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<3xi64>
  // CHECK-SAME: ) -> memref<?xi64> {
  func @concat_op_lowering_mixed_1(%q0 : !qssa.qubit<2>, %q1 : !qssa.qubit<3>) -> !qssa.qubit<?> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[INP1:.*]] = memref_cast %[[ARG1]] : memref<3xi64> to memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = call @__mlir_qssa_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[INP1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    %q2 = qssa.concat %q0, %q1 : (!qssa.qubit<2>, !qssa.qubit<3>) -> !qssa.qubit<?>

    // CHECK: return %[[RES]] : memref<?xi64>
    return %q2 : !qssa.qubit<?>
  }

  // concat (static, dynamic) -> static
  // CHECK-LABEL: func @concat_op_lowering_mixed_2(
  // CHECK-SAME: %[[ARG0:.*]]: memref<2xi64>, %[[ARG1:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> memref<5xi64> {
  func @concat_op_lowering_mixed_2(%q0 : !qssa.qubit<2>, %q1 : !qssa.qubit<?>) -> !qssa.qubit<5> {
    // CHECK: %[[INP0:.*]] = memref_cast %[[ARG0]] : memref<2xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_qssa_simulator__concat_qubits(
    // CHECK-SAME: %[[INP0]], %[[ARG1]]) : (memref<?xi64>, memref<?xi64>) -> memref<?xi64>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi64> to memref<5xi64>
    %q2 = qssa.concat %q0, %q1 : (!qssa.qubit<2>, !qssa.qubit<?>) -> !qssa.qubit<5>

    // CHECK: return %[[RES]] : memref<5xi64>
    return %q2 : !qssa.qubit<5>
  }


  // CHECK-LABEL: func @split_op_lowering_dynamic(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>,
  // CHECK-SAME: %[[SIZE0:.*]]: index,
  // CHECK-SAME: %[[SIZE1:.*]]: index
  // CHECK-SAME: ) -> (memref<?xi64>, memref<?xi64>) {
  func @split_op_lowering_dynamic(%q : !qssa.qubit<?>, %l : index, %r : index)
        -> (!qssa.qubit<?>, !qssa.qubit<?>) {
    // CHECK: %[[OUT:.*]]:2 = call @__mlir_qssa_simulator__split_qubits(
    // CHECK-SAME: %[[ARG]], %[[SIZE0]], %[[SIZE1]]
    // CHECK-SAME: ) : (memref<?xi64>, index, index) -> (memref<?xi64>, memref<?xi64>)
    %q1, %q2 = qssa.split %q[%l, %r] : !qssa.qubit<?> -> (!qssa.qubit<?>, !qssa.qubit<?>)

    // CHECK: %[[OUT]]#0, %[[OUT]]#1 : memref<?xi64>, memref<?xi64>
    return %q1, %q2 : !qssa.qubit<?>, !qssa.qubit<?>
  }

  // CHECK-LABEL: func @split_op_lowering_static(
  // CHECK-SAME: %[[ARG:.*]]: memref<5xi64>
  // CHECK-SAME: ) -> (memref<2xi64>, memref<3xi64>) {
  func @split_op_lowering_static(%q : !qssa.qubit<5>) -> (!qssa.qubit<2>, !qssa.qubit<3>) {
    // CHECK: %[[INP:.*]] = memref_cast %[[ARG]] : memref<5xi64> to memref<?xi64>
    // CHECK-NEXT: %[[SIZE0:.*]] = constant 2 : index
    // CHECK-NEXT: %[[SIZE1:.*]] = constant 3 : index
    // CHECK-NEXT: %[[OUT:.*]]:2 = call @__mlir_qssa_simulator__split_qubits(
    // CHECK-SAME: %[[INP]], %[[SIZE0]], %[[SIZE1]]
    // CHECK-SAME: ) : (memref<?xi64>, index, index) -> (memref<?xi64>, memref<?xi64>)
    // CHECK-NEXT: %[[RES0:.*]] = memref_cast %[[OUT]]#0 : memref<?xi64> to memref<2xi64>
    // CHECK-NEXT: %[[RES1:.*]] = memref_cast %[[OUT]]#1 : memref<?xi64> to memref<3xi64>
    %q1, %q2 = qssa.split %q : !qssa.qubit<5> -> (!qssa.qubit<2>, !qssa.qubit<3>)

    // CHECK: %[[RES0]], %[[RES1]] : memref<2xi64>, memref<3xi64>
    return %q1, %q2 : !qssa.qubit<2>, !qssa.qubit<3>
  }

  // CHECK-LABEL: func @split_op_lowering_mixed(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>
  // CHECK-SAME: %[[SIZE0:.*]]: index
  // CHECK-SAME: ) -> (memref<?xi64>, memref<3xi64>) {
  func @split_op_lowering_mixed(%q : !qssa.qubit<?>, %head: index) -> (!qssa.qubit<?>, !qssa.qubit<3>) {
    // CHECK: %[[SIZE1:.*]] = constant 3 : index
    // CHECK-NEXT: %[[OUT:.*]]:2 = call @__mlir_qssa_simulator__split_qubits(
    // CHECK-SAME: %[[ARG]], %[[SIZE0]], %[[SIZE1]]
    // CHECK-SAME: ) : (memref<?xi64>, index, index) -> (memref<?xi64>, memref<?xi64>)
    // CHECK-NEXT: %[[RES1:.*]] = memref_cast %[[OUT]]#1 : memref<?xi64> to memref<3xi64>
    %q1, %q2 = qssa.split %q[%head] : !qssa.qubit<?> -> (!qssa.qubit<?>, !qssa.qubit<3>)

    // CHECK: return %[[OUT]]#0, %[[RES1]] : memref<?xi64>, memref<3xi64>
    return %q1, %q2 : !qssa.qubit<?>, !qssa.qubit<3>
  }


  // CHECK-LABEL: func @measure_op_lowering_dynamic(
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi64>
  // CHECK-SAME: ) -> i1 {
  func @measure_op_lowering_dynamic(%q0 : !qssa.qubit<?>) -> i1 {
    // CHECK: %[[RES:.*]] = call @__mlir_qssa_simulator__measure_qubits(
    // CHECK-SAME: %[[ARG]]) : (memref<?xi64>) -> memref<?xi1>
    %res = qssa.measure %q0 : !qssa.qubit<?> -> memref<?xi1>

    %0 = constant 0 : index
    // CHECK: %{{.*}} = load %[[RES]][%{{.*}}] : memref<?xi1>
    %v = load %res[%0] : memref<?xi1>
    return %v : i1
  }

  // CHECK-LABEL: func @measure_op_lowering_static(
  // CHECK-SAME: %[[ARG:.*]]: memref<5xi64>
  // CHECK-SAME: ) -> i1 {
  func @measure_op_lowering_static(%q0 : !qssa.qubit<5>) -> i1 {
    // CHECK: %[[INP:.*]] = memref_cast %[[ARG]] : memref<5xi64> to memref<?xi64>
    // CHECK-NEXT: %[[OUT:.*]] = call @__mlir_qssa_simulator__measure_qubits(
    // CHECK-SAME: %[[INP]]) : (memref<?xi64>) -> memref<?xi1>
    // CHECK-NEXT: %[[RES:.*]] = memref_cast %[[OUT]] : memref<?xi1> to memref<5xi1>
    %res = qssa.measure %q0 : !qssa.qubit<5> -> memref<5xi1>

    %0 = constant 0 : index
    // CHECK: %{{.*}} = load %[[RES]][%{{.*}}] : memref<5xi1>
    %v = load %res[%0] : memref<5xi1>
    return %v : i1
  }
}
