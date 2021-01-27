// no-run: quantum-opt --convert-quantum-to-std %s | FileCheck %s

module {
  // CHECK: func @static_argument_conversion(memref<10xi64>)
  func private @static_argument_conversion(!quantum.qubit<10>)

  // CHECK: func @dynamic_argument_conversion(memref<?xi64>)
  func private @dynamic_argument_conversion(!quantum.qubit<?>)

  // CHECK: func @static_result_conversion() -> memref<10xi64>
  func private @static_result_conversion() -> !quantum.qubit<10>

  // CHECK: func @dynamic_result_conversion() -> memref<?xi64>
  func private @dynamic_result_conversion() -> !quantum.qubit<?>

  // CHECK: func @return_conversion(%[[QUBITS:.*]]: memref<?xi64>) -> memref<?xi64>
  func @return_conversion(%q : !quantum.qubit<?>) -> !quantum.qubit<?> {
    // CHECK: return %[[QUBITS:.*]] : memref<?xi64>
    return %q : !quantum.qubit<?>
  }
}
