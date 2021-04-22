// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-scf --convert-qasm-to-qssa %s | quantum-opt | FileCheck %s

func @foo(%0 : !qasm.qubit) attributes {qasm.gate} {
  qasm.gate @z(%0) : (!qasm.qubit) -> ()
  return {qasm.gate_end}
}

// CHECK: func @ifconv
func @ifconv(%val : f64) -> i1 {
  // CHECK: %[[a:.*]] = qssa.alloc : !qssa.qubit<1>
  %0 = qasm.allocate
  // CHECK: %[[b:.*]] = qssa.alloc : !qssa.qubit<1>
  %1 = qasm.allocate
  %cst_0 = constant 0.0 : f64
  qasm.U(%cst_0 : f64, %cst_0 : f64, %cst_0 : f64) %0
  qasm.gate @x(%0) : (!qasm.qubit) -> ()
  qasm.gate @y(%1) : (!qasm.qubit) -> ()
  // CHECK: %[[mem:.*]] = memref.alloc() : memref<1xi1>
  %mem = memref.alloc() : memref<1xi1>
  // CHECK: %[[qs:.*]] = scf.if 
  qasm.if %mem = 0 : memref<1xi1> {
    qasm.CX %0, %1
  }
  qasm.if %mem = 0 : memref<1xi1> {
    qasm.gate @x(%0) : (!qasm.qubit) -> ()
  }
  qasm.if %mem = 0 : memref<1xi1> {
    call @foo(%1) {qasm.gate} : (!qasm.qubit) -> ()
  }
  %sum = addf %val, %val : f64
  qasm.if %mem = 0 : memref<1xi1> {
    qasm.U(%sum : f64, %sum : f64, %sum : f64) %0
  }
  qasm.CX %0, %1
  %res = qasm.measure %0
  return %res : i1
}

