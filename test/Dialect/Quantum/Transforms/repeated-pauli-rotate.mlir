// RUN: quantum-opt %s | quantum-opt --canonicalize
func @repeatedRx(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %alpha = constant 1.0 : f64
  %beta = constant 2.0 : f64
  %1 = qssa.Rx(%alpha : f64) %0 : !qssa.qubit<1>
  %2 = qssa.Rx(%beta : f64) %1 : !qssa.qubit<1>
  return %2 : !qssa.qubit<1>
}
func @repeatedRy(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %alpha = constant 1.0 : f64
  %beta = constant 2.0 : f64
  %1 = qssa.Ry(%alpha : f64) %0 : !qssa.qubit<1>
  %2 = qssa.Ry(%beta : f64) %1 : !qssa.qubit<1>
  return %2 : !qssa.qubit<1>
}

func @onlyID(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = qssa.id %0 : !qssa.qubit<1>
  %2 = qssa.id %1 : !qssa.qubit<1>
  return %2 : !qssa.qubit<1>
}

func @callRxRx(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = call @repeatedRx(%0) : (!qssa.qubit<1>) -> !qssa.qubit<1>
  return %1 : !qssa.qubit<1>
}
