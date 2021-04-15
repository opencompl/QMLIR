// RUN: quantum-opt --qssa-apply-rewrites %s | FileCheck %s

// CHECK: func @removephase(%[[q:.*]]: !qssa.qubit<1>) -> tensor<1xi1> {
func @removephase(%0 : !qssa.qubit<1>) -> tensor<1xi1> {
  %cst = constant 1.0 : f64
  %1 = qssa.Rz(%cst : f64) %0 : !qssa.qubit<1>
  %2 = qssa.T %1 : !qssa.qubit<1>
  %3 = qssa.S %2 : !qssa.qubit<1>
  %4 = qssa.Z %3 : !qssa.qubit<1>
  // CHECK: %[[res:.*]], %[[q1:.*]] = qssa.measure %[[q]] : !qssa.qubit<1> -> tensor<1xi1>
  %res, %5 = qssa.measure %4 : !qssa.qubit<1> -> tensor<1xi1>
  // CHECK: return %[[res]] : tensor<1xi1>
  return %res : tensor<1xi1>
}

// CHECK: func @removephaseOne(%[[q:.*]]: !qssa.qubit<1>) -> i1 {
func @removephaseOne(%0 : !qssa.qubit<1>) -> i1 {
  %cst = constant 1.0 : f64
  %1 = qssa.Rz(%cst : f64) %0 : !qssa.qubit<1>
  %2 = qssa.T %1 : !qssa.qubit<1>
  %3 = qssa.S %2 : !qssa.qubit<1>
  %4 = qssa.Z %3 : !qssa.qubit<1>
  // CHECK: %[[res:.*]], %[[q1:.*]] = qssa.measure_one %[[q]]
  %res, %5 = qssa.measure_one %4
  // CHECK: return %[[res]] : i1
  return %res : i1
}
