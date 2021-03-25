// RUN: quantum-opt %s
// RUN: quantum-opt %s --convert-qasm-to-qssa

func @barrier() {
  %mem = memref.alloc() : memref<2xi1>
  %0 = qasm.allocate
  %1 = qasm.allocate
  qasm.CX %0, %1
  qasm.barrier %0
  qasm.CX %0, %1
  %res = qasm.measure %0
  affine.store %res, %mem[0] : memref<2xi1>
  return
}

func @nobarrier() {
  %mem = memref.alloc() : memref<2xi1>
  %0 = qasm.allocate
  %1 = qasm.allocate
  qasm.CX %0, %1
  qasm.CX %0, %1
  %res = qasm.measure %0
  affine.store %res, %mem[0] : memref<2xi1>
  return
}
