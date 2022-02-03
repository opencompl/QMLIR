// RUN: quantum-opt %s 2>&1 | FileCheck %s

// CHECK: func
func @iftest() {
  %0 = qasm.allocate
  %1 = qasm.allocate
  %creg = memref.alloc() : memref<10xi1>
  qasm.if %creg = 0 : memref<10xi1> {
    qasm.CX %0, %1
  }
  return
}

