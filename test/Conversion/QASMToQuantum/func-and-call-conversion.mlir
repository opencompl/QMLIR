// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt | FileCheck %s

// CHECK: func private @somegate(%[[cst:.*]]: f32, %[[q0:.*]]: !qssa.qubit<1>) 
// CHECK: -> !qssa.qubit<1> attributes {qasm.stdgate = "cx"} {
func private @somegate(%cst : f32, %0 : !qasm.qubit) attributes {qasm.gate, qasm.stdgate="cx"} {
  // CHECK-NEXT: %[[q1:.*]] = qssa.U(%[[cst]] : f32, %[[cst]] : f32, %[[cst]] : f32) %[[q0]]
  qasm.U(%cst : f32, %cst : f32, %cst : f32) %0
  // CHECK-NEXT: %[[q2:.*]] = qssa.U(%[[cst]] : f32, %[[cst]] : f32, %[[cst]] : f32) %[[q1]]
  qasm.U(%cst : f32, %cst : f32, %cst : f32) %0
  // CHECK-NEXT: return %[[q2]] : !qssa.qubit<1>
  return {qasm.gate_end}
  // CHECK-NEXT: }
}

// CHECK: func @callconv() {
func @callconv() {
  // CHECK-NEXT: %[[cst:.*]] = constant [[zero:.*]] : f32
  %cst = constant 0.0 : f32
  // CHECK-NEXT: %[[q0:.*]] = qssa.alloc : !qssa.qubit<1>
  %0 = qasm.allocate
  // CHECK-NEXT: %[[q1:.*]] = call @somegate(%[[cst]], %[[q0]]) : (f32, !qssa.qubit<1>) -> !qssa.qubit<1>
  call @somegate(%cst, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
  // CHECK-NEXT: %[[q2:.*]] = call @somegate(%[[cst]], %[[q1]]) : (f32, !qssa.qubit<1>) -> !qssa.qubit<1>
  call @somegate(%cst, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
  // CHECK-NEXT: return
  return
  // CHECK-NEXT: }
}
