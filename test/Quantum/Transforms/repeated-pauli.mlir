// RUN: quantum-opt %s | quantum-opt --canonicalize
func @repeatedXandY(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = qssa.X %0 : !qssa.qubit<1>
  %2 = qssa.X %1 : !qssa.qubit<1>
  %3 = qssa.Y %2 : !qssa.qubit<1>
  %4 = qssa.Y %3 : !qssa.qubit<1>
  return %4 : !qssa.qubit<1>
}

func @onlyID(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = qssa.id %0 : !qssa.qubit<1>
  %2 = qssa.id %1 : !qssa.qubit<1>
  return %2 : !qssa.qubit<1>
}

func @callID(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = call @repeatedXandY(%0) : (!qssa.qubit<1>) -> !qssa.qubit<1>
  return %1 : !qssa.qubit<1>
}
