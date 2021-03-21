// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt

func @cxconv() {
  %0 = qasm.allocate
  %1 = qasm.allocate
  qasm.CX %0, %1
  qasm.CX %0, %1
  return
}
