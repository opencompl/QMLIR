// RUN: quantum-opt %s
// RUN: quantum-opt %s --qssa-gate-count

func private @bar(%0 : !qssa.qubit<2>) -> !qssa.qubit<2> {
  %1 = qssa.T %0 : !qssa.qubit<2>
  %2 = qssa.S %1 : !qssa.qubit<2>
  %3 = qssa.T %2 : !qssa.qubit<2>
  return %3 : !qssa.qubit<2>
}
func private @foo(%0 : !qssa.qubit<2>) -> !qssa.qubit<2> {
  %1 = qssa.T %0 : !qssa.qubit<2>
  %2 = qssa.S %1 : !qssa.qubit<2>
  %3 = call @bar(%2) : (!qssa.qubit<2>) -> (!qssa.qubit<2>)
  %4 = qssa.T %3 : !qssa.qubit<2>
  return %4 : !qssa.qubit<2>
}
func @main() {
  %0 = qssa.alloc : !qssa.qubit<2>
  %1 = call @foo(%0) : (!qssa.qubit<2>) -> (!qssa.qubit<2>)
  %2 = call @foo(%1) : (!qssa.qubit<2>) -> (!qssa.qubit<2>)
  qssa.sink %2 : !qssa.qubit<2>
  return
}
