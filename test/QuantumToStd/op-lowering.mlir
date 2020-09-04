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
}
