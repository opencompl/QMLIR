// RUN: quantum-opt %s | quantum-opt 

func @toffoli(%in: !quantum.qubit<3>) -> !quantum.qubit<3> {
//  %c0, %q0 = quantum.split %in : !quantum.qubit<3> -> (!quantum.qubit<2>, !quantum.qubit<1>)
//  %c1, %q1 = quantum.controlled [%c0 : !quantum.qubit<2>] %q0 : !quantum.qubit<1>
//  %out = quantum.concat %c1, %q1 : (!quantum.qubit<2>, !quantum.qubit<1>) -> !quantum.qubit<3>
//  return %out: !quantum.qubit<3>
  return %in: !quantum.qubit<3>
}

func @main() {
  %q0 = quantum.allocate() : !quantum.qubit<3>
  %q1 = call @toffoli(%q0) : (!quantum.qubit<3>) -> !quantum.qubit<3>
  %q2, %q3 = quantum.split %q1 : !quantum.qubit<3> -> (!quantum.qubit<1>, !quantum.qubit<2>)
  %res = quantum.measure %q2 : !quantum.qubit<1> -> memref<1xi1>

  %idx0 = constant 0 : index
  %r0 = load %res[%idx0] : memref<1xi1>

  %q4 = scf.if %r0 -> !quantum.qubit<2> {
    %q5, %q6 = quantum.split %q3 : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q7 = quantum.concat %q6, %q5 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
    scf.yield %q7 : !quantum.qubit<2>
  } else {
    scf.yield %q3 : !quantum.qubit<2>
  }

  %res1 = quantum.measure %q4 : !quantum.qubit<2> -> memref<2xi1>

  return
}
