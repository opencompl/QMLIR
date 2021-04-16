// RUN: quantum-opt %s
// RUN: quantum-opt %s --qssa-compute-depths

func @main(%b : i1) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.H %0 : !qssa.qubit<1>
  %2 = qssa.barrier %1 : !qssa.qubit<1>
  %3 = qssa.X %2 : !qssa.qubit<1>
  %4 = scf.if %b -> (!qssa.qubit<1>) {
    %l = qssa.Y %3 : !qssa.qubit<1>
    scf.yield %l : !qssa.qubit<1>
  } else {
    scf.yield %3 : !qssa.qubit<1>
  }
  %5 = scf.if %b -> (!qssa.qubit<1>) {
    %l = qssa.Y %4 : !qssa.qubit<1>
    scf.yield %l : !qssa.qubit<1>
  } else {
    scf.yield %4 : !qssa.qubit<1>
  }
  %6 = scf.if %b -> (!qssa.qubit<1>) {
    %l = qssa.Y %5 : !qssa.qubit<1>
    scf.yield %l : !qssa.qubit<1>
  } else {
    scf.yield %5 : !qssa.qubit<1>
  }
  %7 = qssa.X %6 : !qssa.qubit<1>
  qssa.sink %7 : !qssa.qubit<1>
  return
}
