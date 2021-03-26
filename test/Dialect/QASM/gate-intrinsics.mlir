// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt %s --inline --symbol-dce

func private @cx(%0 : !qasm.qubit, %1: !qasm.qubit) attributes {qasm.gate, qasm.stdgate="cx"} {
  qasm.CX %0, %1
  return
}

func private @id() attributes {matrix=dense<[[1.0,0.0],[0.0,1.0]]>:tensor<2x2xf32>}

func @main() {
  %0 = qasm.allocate
  %1 = qasm.allocate
  call @cx(%0, %1) : (!qasm.qubit, !qasm.qubit) -> ()
  qasm.gate @cx(%0, %1) : (!qasm.qubit, !qasm.qubit) -> ()
  qasm.gate @id(%0) : (!qasm.qubit) -> ()
  return
}
