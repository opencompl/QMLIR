// RUN: quantum-opt %s --qssa-convert-1q-to-U

func @foo(%cst : f64) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.X %0 : !qssa.qubit<1>
  %2 = qssa.Y %1 : !qssa.qubit<1>
  %3 = qssa.Z %2 : !qssa.qubit<1>
  %4 = qssa.T %3 : !qssa.qubit<1>
  %5 = qssa.S %4 : !qssa.qubit<1>
  %6 = qssa.Tdg %5 : !qssa.qubit<1>
  %7 = qssa.Sdg %6 : !qssa.qubit<1>
  %8 = qssa.Rx(%cst : f64) %7 : !qssa.qubit<1>
  %9 = qssa.Ry(%cst : f64) %8 : !qssa.qubit<1>
  %10 = qssa.Rz(%cst : f64) %9 : !qssa.qubit<1>
  qssa.sink %10 : !qssa.qubit<1>
  return
}


