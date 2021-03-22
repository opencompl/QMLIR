// RUN: quantum-opt %s
// RUN: quantum-opt --convert-qasm-to-qssa %s | quantum-opt

func @myg (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate} {
  %2 = std.constant 0 : i32
  %3 = std.sitofp %2 : i32 to f32
  %4 = std.constant 0 : i32
  %5 = std.sitofp %4 : i32 to f32
  qasm.U (%0 : f32, %3 : f32, %5 : f32) %1
  std.return {qasm.gate_end}
}
func @qasm_main () -> ()  {
  %0 = qasm.allocate
  %1 = qasm.allocate
  %2 = std.constant 1 : i32
  %3 = std.constant 0 : i32
  %4 = std.subi %3, %2 : i32
  %5 = std.sitofp %4 : i32 to f32
  std.call @myg(%5, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
  std.call @myg(%5, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
  %6 = qasm.allocate
  %7 = qasm.allocate
  %8 = memref.alloc () : memref<2xi1>
  %9 = qasm.measure %6
  %10 = std.constant 0 : index
  memref.store %9, %8[%10] : memref<2xi1>
  %11 = qasm.measure %7
  %12 = std.constant 1 : index
  memref.store %11, %8[%12] : memref<2xi1>
  %13 = qasm.measure %6
  %14 = std.constant 0 : index
  memref.store %13, %8[%14] : memref<2xi1>
  qasm.reset %1
  std.return
}
