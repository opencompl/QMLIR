// RUN: quantum-opt %s | quantum-opt
func @cx(%0: !qasm.qubit, %1: !qasm.qubit) attributes { qasm.gate, qasm.stdgate.cx } {
  qasm.CX %0, %1
  return
}

