// RUN: quantum-opt %s | quantum-opt 

#matX = dense<[[0.0, 1.0], [1.0, 0.0]]> : tensor<2x2xf64>
#vec = dense<[1,2,3]> : tensor<3xi32>
#gateX = {
  name = "X",
  size = 1,
  matrix = #matX
}

#toffoli = {
  name = "Toffoli",
  size = 3,
  matrix = sparse<
    [[0, 0], [1, 1], [2, 2], [3, 3],
     [4, 4], [5, 5], [6, 7], [7, 6]],
    [ 1.0,    1.0,    1.0,    1.0,
      1.0,    1.0,    1.0,    1.0]> : tensor<8x8xf32>
}

func @main() {
  %q0 = quantum.allocate : !quantum.qubit<3>
  %q1 = quantum.transform #toffoli(%q0) : !quantum.qubit<3>
  %q2, %q3 = quantum.split %q1 : !quantum.qubit<3> -> (!quantum.qubit<1>, !quantum.qubit<2>)
  %res = quantum.measure %q2 : !quantum.qubit<1> -> tensor<1xi1>

  %idx0 = constant 0 : index
  %r0 = extract_element %res[%idx0] : tensor<1xi1>

  %q4 = scf.if %r0 -> !quantum.qubit<2> {
    %q5, %q6 = quantum.split %q3 : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q7 = quantum.concat %q6, %q5 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
    scf.yield %q7 : !quantum.qubit<2>
  } else {
    scf.yield %q3 : !quantum.qubit<2>
  }

  %res1 = quantum.measure %q4 : !quantum.qubit<2> -> tensor<2xi1>

  return
}
