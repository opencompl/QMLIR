// no-run: quantum-opt --convert-qssa-to-qir %s | FileCheck %s

module {
  // CHECK: func @static_argument_conversion(memref<10xi64>)
  func private @static_argument_conversion(!qssa.qubit<10>)

  // CHECK: func @dynamic_argument_conversion(memref<?xi64>)
  func private @dynamic_argument_conversion(!qssa.qubit<?>)

  // CHECK: func @static_result_conversion() -> memref<10xi64>
  func private @static_result_conversion() -> !qssa.qubit<10>

  // CHECK: func @dynamic_result_conversion() -> memref<?xi64>
  func private @dynamic_result_conversion() -> !qssa.qubit<?>

  // CHECK: func @return_conversion(%[[QUBITS:.*]]: memref<?xi64>) -> memref<?xi64>
  func @return_conversion(%q : !qssa.qubit<?>) -> !qssa.qubit<?> {
    // CHECK: return %[[QUBITS:.*]] : memref<?xi64>
    return %q : !qssa.qubit<?>
  }
}
