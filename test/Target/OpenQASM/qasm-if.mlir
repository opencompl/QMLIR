// RUN: quantum-translate --mlir-to-openqasm %s

func private @cx (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate="cx"}
{
  %1 = std.constant 0 : i32
  %2 = std.sitofp %1 : i32 to f32
  %3 = std.constant 0 : i32
  %4 = std.sitofp %3 : i32 to f32
  %5 = std.constant 0 : i32
  %6 = std.sitofp %5 : i32 to f32
  qasm.U (%2 : f32, %4 : f32, %6 : f32) %0
  std.return {qasm.gate_end}
}
func private @myg (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate}
{
  %2 = std.constant 0 : i32
  %3 = std.sitofp %2 : i32 to f32
  %4 = std.constant 0 : i32
  %5 = std.sitofp %4 : i32 to f32
  qasm.U (%0 : f32, %3 : f32, %5 : f32) %1
  std.return {qasm.gate_end}
}
func private @myop (%0 : f32, %1 : f32, %2 : !qasm.qubit, %3 : !qasm.qubit) -> () attributes {qasm.gate}
func  @qasm_main () -> () attributes {qasm.main}
{
  %0 = qasm.allocate 
  %1 = qasm.allocate 
  %2 = qasm.allocate 
  %3 = qasm.allocate 
  %4 = memref.alloc () : memref<4xi1>
  %5 = qasm.allocate 
  %6 = std.constant 1 : i32
  %7 = std.constant 0 : i32
  %8 = std.subi %7, %6 : i32
  %9 = std.sitofp %8 : i32 to f32
  std.call @myg(%9, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
  std.call @myg(%9, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
  std.call @myg(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
  std.call @myg(%9, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
  %10 = qasm.measure %0
  %11 = std.constant 0 : index
  affine.store %10, %4[%11] : memref<4xi1>
  %12 = qasm.measure %1
  %13 = std.constant 1 : index
  affine.store %12, %4[%13] : memref<4xi1>
  %14 = qasm.measure %2
  %15 = std.constant 2 : index
  affine.store %14, %4[%15] : memref<4xi1>
  %16 = qasm.measure %3
  %17 = std.constant 3 : index
  affine.store %16, %4[%17] : memref<4xi1>
  qasm.barrier %0
  qasm.barrier %1
  qasm.barrier %2
  qasm.barrier %3
  qasm.barrier %5
  qasm.reset %0
  qasm.reset %1
  qasm.reset %2
  qasm.reset %3
  qasm.CX %0, %5
  qasm.CX %1, %5
  qasm.CX %2, %5
  qasm.CX %3, %5
  %18 = std.constant 0 : i32
  %19 = std.constant 0 : i32
  %20 = std.addi %18, %19 : i32
  %21 = std.sitofp %20 : i32 to f32
  %22 = std.constant 0 : i32
  %23 = std.sitofp %22 : i32 to f32
  %24 = std.constant 0 : i32
  %25 = std.sitofp %24 : i32 to f32
  qasm.U (%21 : f32, %23 : f32, %25 : f32) %0
  qasm.if %4 = 15 : memref<4xi1> {
    %26 = std.constant 0 : i32
    %27 = std.sitofp %26 : i32 to f32
    %28 = std.constant 0 : i32
    %29 = std.sitofp %28 : i32 to f32
    %30 = std.constant 0 : i32
    %31 = std.sitofp %30 : i32 to f32
    qasm.U (%27 : f32, %29 : f32, %31 : f32) %0
    qasm.U (%27 : f32, %29 : f32, %31 : f32) %1
    qasm.U (%27 : f32, %29 : f32, %31 : f32) %2
    qasm.U (%27 : f32, %29 : f32, %31 : f32) %3
  }
  std.return 
}
