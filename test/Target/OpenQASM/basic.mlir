// RUN: quantum-translate %s --mlir-to-openqasm

func private @cx() attributes {qasm.stdgate="cx"}

func @some_gate(%a: f32, %q:!qasm.qubit) attributes {qasm.gate} {
  qasm.U(%a : f32, %a : f32, %a : f32) %q
  return
}

func @qasm_main() attributes {qasm.main} {
  %x = qasm.pi : f32
  %y = constant 1.234 : f32
  %z = addf %x, %y : f32
  %0 = qasm.allocate
  %1 = qasm.allocate
  %c = memref.alloc() : memref<16xi1>
  qasm.CX %0, %1
  qasm.CX %1, %0
  qasm.U(%x : f32, %y : f32, %z : f32) %0
  %res = qasm.measure %1
  %idx = constant 4 : index
  memref.store %res, %c[%idx] : memref<16xi1>
  return
}
