// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt

func @somegate(%cst : f32, %0 : !qasm.qubit) attributes {qasm.gate} {
  qasm.U(%cst:f32, %cst:f32, %cst:f32) %0
  // qasm.U(%cst:f32, %cst:f32, %cst:f32) %0
  return {qasm.gate_end}
}

// func @callconv() {
//   %0 = qasm.allocate
//   %cst = constant 0.0 : f32
//   call @somegate(%cst, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
//   return
// }
