// RUN: quantum-opt %s
// RUN: quantum-opt %s --inline --canonicalize | FileCheck %s


// CHECK: func @concat_fold(%[[arg0:.*]]: !qssa.qubit<1>, %[[arg1:.*]]: !qssa.qubit<1>) -> !qssa.qubit<2> {
func @concat_fold(%0: !qssa.qubit<1>, %1: !qssa.qubit<1>) -> !qssa.qubit<2> {
  // CHECK: %[[res:.*]] = qssa.concat %[[arg0]], %[[arg1]] : (!qssa.qubit<1>, !qssa.qubit<1>) -> !qssa.qubit<2>
  %2 = qssa.concat %0, %1 : (!qssa.qubit<1>, !qssa.qubit<1>) -> (!qssa.qubit<2>)
  %3, %4 = qssa.split %2 : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %5 = qssa.concat %3, %4 : (!qssa.qubit<1>, !qssa.qubit<1>) -> (!qssa.qubit<2>)
  // CHECK: return %[[res]] : !qssa.qubit<2>
  return %5 : !qssa.qubit<2>
  // CHECK: }
}

// CHECK: func @split_fold(%[[arg0:.*]]: !qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>) {
func @split_fold(%0: !qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>) {
  // CHECK: %[[res:.*]]:2 = qssa.split %[[arg0]] : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %1, %2 = qssa.split %0 : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %3 = qssa.concat %1, %2 : (!qssa.qubit<1>, !qssa.qubit<1>) -> (!qssa.qubit<2>)
  %4, %5 = qssa.split %3 : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  // CHECK: return %[[res]]#0, %[[res]]#1 : !qssa.qubit<1>, !qssa.qubit<1>
  return %4, %5 : !qssa.qubit<1>, !qssa.qubit<1>
  // CHECK: }
}

// CHECK: func @inline_fold() -> tensor<2xi1> {
func @inline_fold() -> tensor<2xi1> {
  // CHECK: %[[q:.*]] = qssa.alloc : !qssa.qubit<2>
  %0 = qssa.alloc : !qssa.qubit<2>
  %1, %2 = call @split_fold(%0) : (!qssa.qubit<2>) -> (!qssa.qubit<1>, !qssa.qubit<1>)
  %3 = call @concat_fold(%1, %2) : (!qssa.qubit<1>, !qssa.qubit<1>) -> (!qssa.qubit<2>)
  // CHECK: %[[res:.*]], %[[qq:.*]] = qssa.measure %[[q]] : !qssa.qubit<2> -> tensor<2xi1>
  %res, %4 = qssa.measure %3 : !qssa.qubit<2> -> tensor<2xi1>
  // CHECK: return %[[res]] : tensor<2xi1>
  return %res : tensor<2xi1>
  // CHECK: }
}
