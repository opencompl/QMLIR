// RUN: quantum-opt --convert-qssa-to-qir %s | quantum-opt | FileCheck %s

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
