// RUN: quantum-opt --inline --symbol-dce %s
func private @nothing(%0: !qasm.qubit) attributes {qasm.gate} {
  return
}

func @test() {
  // qreg q[2];
  %q0 = qasm.allocate
  %q1 = qasm.allocate
  // creg c[2];
  %c = memref.alloc() : memref<2xi1>
  // if (c == 0) nothing q
  qasm.if %c = 0 : memref<2xi1> {
    call @nothing(%q0) {qasm.gate} : (!qasm.qubit) -> ()
    call @nothing(%q1) {qasm.gate} : (!qasm.qubit) -> ()
  }
  qasm.if %c = 0 : memref<2xi1> {
    qasm.CX %q0, %q1
  }
  qasm.if %c = 0 : memref<2xi1> {
  }
  return
}
