// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt

func @cxconv() {
  %0 = qasm.allocate
  %1 = qasm.allocate
  qasm.CX %0, %1
  qasm.CX %0, %1
  return
}

func @uconv() {
  %cst = constant 0.0 : f32
  %0 = qasm.allocate
  qasm.U(%cst : f32, %cst : f32, %cst : f32) %0
  qasm.U(%cst : f32, %cst : f32, %cst : f32) %0
  return
}
