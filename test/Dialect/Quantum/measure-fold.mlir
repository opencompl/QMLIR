// RUN: quantum-opt %s
// RUN: quantum-opt %s --inline --canonicalize --memref-dataflow-opt | FileCheck %s


// CHECK: func @measure_fold(%[[arg:.*]]: !qssa.qubit<1>) -> i1 {
func @measure_fold(%0: !qssa.qubit<1>) -> i1 {
  // CHECK: %[[res:.*]], %[[q:.*]] = qssa.measure_one %[[arg]]
  %res, %q = qssa.measure_one %0
  // CHECK: return %[[res]] : i1
  return %res : i1
  // CHECK: }
}

// CHECK: func @inline_fold() -> i1 {
func @inline_fold() -> i1 {
  %0 = constant 0 : index
  // CHECK: %[[q:.*]] = qssa.alloc : !qssa.qubit<1>
  %q = qssa.alloc : !qssa.qubit<1>
  // CHECK: %[[r:.*]], %[[q1:.*]] = qssa.measure_one %[[q]]
  %res = call @measure_fold(%q) : (!qssa.qubit<1>) -> i1
  %mem = memref.alloc() : memref<1xi1>
  affine.store %res, %mem[%0] : memref<1xi1>
  %val = affine.load %mem[%0] : memref<1xi1>
  // CHECK: return %[[r]] : i1
  return %val : i1
  // CHECK: }
}
