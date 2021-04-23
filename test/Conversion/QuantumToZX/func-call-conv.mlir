// RUN: quantum-opt %s --qssa-prepare-for-zx --convert-qssa-to-zx

func @foo(%0 : !qssa.qubit<1>) -> !qssa.qubit<1> {
  %1 = qssa.X %0 : !qssa.qubit<1>
  %2 = qssa.Z %1 : !qssa.qubit<1>
  return %2 : !qssa.qubit<1>
}
func @bar() {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = call @foo(%0) : (!qssa.qubit<1>) -> !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @ret() -> !qssa.qubit<1> {
  %0 = qssa.alloc : !qssa.qubit<1>
  return %0 : !qssa.qubit<1>
}
