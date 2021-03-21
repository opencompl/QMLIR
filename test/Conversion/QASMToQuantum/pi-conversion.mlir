// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt | FileCheck %s

func @piconv() {
  // CHECK: %[[_:.*]] = constant [[_:.*]] : f32
  %pi = qasm.pi : f32
  // CHECK: %[[_:.*]] = constant [[_:.*]] : f64
  %pi2 = qasm.pi : f64
  return
}
