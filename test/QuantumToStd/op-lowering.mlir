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
}
