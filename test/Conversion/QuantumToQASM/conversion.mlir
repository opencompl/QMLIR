// RUN: quantum-opt --convert-qssa-to-qasm %s

func @main() -> tensor<2xi1> {
  %qs = qssa.alloc : !qssa.qubit<3>
  %a0, %bc0 = qssa.split %qs : (!qssa.qubit<3>) -> (!qssa.qubit<1>, !qssa.qubit<2>)
  %b0, %c0 = qssa.split %bc0 : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %b1, %c1 = qssa.CNOT %b0, %c0
  %bc1 = qssa.concat %b1, %c1 : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  %bc2 = qssa.X %bc1 : !qssa.qubit<2>
  %bc3 = qssa.Y %bc2 : !qssa.qubit<2>
  %res, %bc4 = qssa.measure %bc3 : !qssa.qubit<2> -> tensor<2xi1>
  return %res : tensor<2xi1>
}
