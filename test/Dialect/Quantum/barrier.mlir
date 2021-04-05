// RUN: quantum-opt %s
// RUN: quantum-opt %s --canonicalize | FileCheck %s


// CHECK: func @barrier() {
func @barrier() {
  // CHECK-NEXT: %[[q0:.*]] = qssa.alloc : !qssa.qubit<10>
  %0 = qssa.alloc : !qssa.qubit<10>
  // CHECK-NEXT: %[[q1:.*]] = qssa.X %[[q0]] : !qssa.qubit<10>
  %1 = qssa.X %0 : !qssa.qubit<10>
  // CHECK-NEXT: %[[q2:.*]] = qssa.barrier %[[q1]] : !qssa.qubit<10>
  %2 = qssa.barrier %1 : !qssa.qubit<10>
  // CHECK-NEXT: %[[q3:.*]] = qssa.X %[[q2]] : !qssa.qubit<10>
  %3 = qssa.X %2 : !qssa.qubit<10>
  // CHECK-NEXT: qssa.sink %[[q3]] : !qssa.qubit<10>
  qssa.sink %3 : !qssa.qubit<10>
  // CHECK-NEXT: return
  return
// CHECK-NEXT: }
}

// CHECK-NEXT: func @nobarrier() {
func @nobarrier() {
  // CHECK-NEXT: %[[q0:.*]] = qssa.alloc : !qssa.qubit<10>
  %0 = qssa.alloc : !qssa.qubit<10>
  %1 = qssa.X %0 : !qssa.qubit<10>
  %2 = qssa.X %1 : !qssa.qubit<10>
  // CHECK-NEXT: qssa.sink %[[q0]] : !qssa.qubit<10>
  qssa.sink %2 : !qssa.qubit<10>
  // CHECK-NEXT: return
  return
// CHECK-NEXT: }
}
