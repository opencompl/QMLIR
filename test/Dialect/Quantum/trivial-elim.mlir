// no-run: quantum-opt %s --qssa-check-single-use
// RUN: quantum-opt %s --cse --inline --symbol-dce --canonicalize | quantum-opt | FileCheck %s

// Code that gets trivially optimized out fully

func private @cx(%qs : !qssa.qubit<2>) -> !qssa.qubit<2> {
  %a, %b = qssa.split %qs : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %a1, %b1 = qssa.CNOT %a, %b
  %qs1 = qssa.concat %a1, %b1 : (!qssa.qubit<1>, !qssa.qubit<1>) -> (!qssa.qubit<2>)
  return %qs1 : !qssa.qubit<2>
}

// CHECK: func @main() {
func @main() {
  %qs = qssa.alloc : !qssa.qubit<2>
  %qs1 = call @cx(%qs) : (!qssa.qubit<2>) -> !qssa.qubit<2>
// CHECK-NEXT:   return
  return
// CHECK-NEXT: }
}
