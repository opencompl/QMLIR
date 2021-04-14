// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt | FileCheck %s

func @ifconv() {
  %0 = qasm.allocate
  %1 = qasm.allocate
  %mem = memref.alloc() : memref<3xi1>
  qasm.if %mem = 0 : memref<3xi1> {
    qasm.CX %0, %1
  }
  return
}

