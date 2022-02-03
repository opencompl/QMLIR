// RUN: quantum-translate --mlir-to-openqasm %s

module  {
  func @qasm_main() attributes {qasm.main} {
    %c0_i32 = constant 0 : i32
    %0 = qasm.allocate
    %1 = qasm.allocate
    %2 = qasm.allocate
    %3 = qasm.allocate
    %4 = memref.alloc() : memref<4xi1>
    %5 = qasm.measure %0
    affine.store %5, %4[0] : memref<4xi1>
    %6 = qasm.measure %1
    affine.store %6, %4[1] : memref<4xi1>
    %7 = qasm.measure %2
    affine.store %7, %4[2] : memref<4xi1>
    %8 = qasm.measure %3
    affine.store %8, %4[3] : memref<4xi1>
    qasm.if %4 = 0 : memref<4xi1>  {
      %9 = sitofp %c0_i32 : i32 to f32
      qasm.U(%9 : f32, %9 : f32, %9 : f32) %0
      qasm.U(%9 : f32, %9 : f32, %9 : f32) %1
      qasm.U(%9 : f32, %9 : f32, %9 : f32) %2
      qasm.U(%9 : f32, %9 : f32, %9 : f32) %3
    }
    return
  }
}

