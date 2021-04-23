// RUN: quantum-opt %s --qssa-prepare-for-zx --convert-qssa-to-zx

func @xnode() {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.X %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @rxnode(%cst : f64) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.Rx(%cst : f64) %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @rxrx(%cst : f64) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.Rx(%cst : f64) %0 : !qssa.qubit<1>
  %2 = qssa.Rx(%cst : f64) %1 : !qssa.qubit<1>
  qssa.sink %2 : !qssa.qubit<1>
  return
}

func @znode() {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.Z %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}
func @rznode(%cst : f64) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.Rz(%cst : f64) %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}
