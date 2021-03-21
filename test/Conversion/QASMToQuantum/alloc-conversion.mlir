// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt | FileCheck %s

func @alloc_conversion() {
  // CHECK: %[[_:.*]] = qssa.allocate() : !qssa.qubit<1>
  %0 = qasm.allocate
  // CHECK: %[[_:.*]] = qssa.allocate() : !qssa.qubit<1>
  %1 = qasm.allocate
  return
}
