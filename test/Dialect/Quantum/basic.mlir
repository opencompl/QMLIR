// RUN: quantum-opt %s | quantum-opt 

func @toffoli(%in: !qssa.qubit<3>) -> !qssa.qubit<3> {
//  %c0, %q0 = qssa.split %in : !qssa.qubit<3> -> (!qssa.qubit<2>, !qssa.qubit<1>)
//  %c1, %q1 = qssa.controlled [%c0 : !qssa.qubit<2>] %q0 : !qssa.qubit<1>
//  %out = qssa.concat %c1, %q1 : (!qssa.qubit<2>, !qssa.qubit<1>) -> !qssa.qubit<3>
//  return %out: !qssa.qubit<3>
  return %in: !qssa.qubit<3>
}

func @main() -> tensor<2xi1> {
  %q0 = qssa.alloc() : !qssa.qubit<3>
  %q1 = call @toffoli(%q0) : (!qssa.qubit<3>) -> !qssa.qubit<3>
  %q2, %q3 = qssa.split %q1 : (!qssa.qubit<3>) -> (!qssa.qubit<1>, !qssa.qubit<2>)
  %res, %ign0 = qssa.measure %q2 : !qssa.qubit<1> -> tensor<1xi1>

  %idx0 = constant 0 : index
  %r0 = tensor.extract %res[%idx0] : tensor<1xi1>

  %q4 = scf.if %r0 -> !qssa.qubit<2> {
    %q5, %q6 = qssa.split %q3 : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
    %q7 = qssa.concat %q6, %q5 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
    scf.yield %q7 : !qssa.qubit<2>
  } else {
    scf.yield %q3 : !qssa.qubit<2>
  }

  %res1, %ign1 = qssa.measure %q4 : !qssa.qubit<2> -> tensor<2xi1>

  return %res1 : tensor<2xi1>
}
