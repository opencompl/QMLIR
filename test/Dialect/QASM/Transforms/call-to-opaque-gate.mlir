// RUN: quantum-opt --inline %s
// RUN: quantum-opt --qasm-make-gates-opaque="gates=x,y,z" --inline %s

func private @u3(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "u3"} {
  qasm.U(%arg0 : f64, %arg1 : f64, %arg2 : f64) %arg3
  return {qasm.gate_end}
}
func private @u1(%arg0: f64, %arg1: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "u1"} {
  %cst = constant 0.000000e+00 : f64
  qasm.U(%cst : f64, %cst : f64, %arg0 : f64) %arg1
  return {qasm.gate_end}
}
func private @x(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "x"} {
  %cst = constant 3.1415926535897931 : f64
  %cst_0 = constant 0.000000e+00 : f64
  call @u3(%cst, %cst_0, %cst, %arg0) {qasm.gate} : (f64, f64, f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @y(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "y"} {
  %cst = constant 1.5707963267948966 : f64
  %cst_0 = constant 3.1415926535897931 : f64
  call @u3(%cst_0, %cst, %cst, %arg0) {qasm.gate} : (f64, f64, f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @z(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "z"} {
  %cst = constant 3.1415926535897931 : f64
  call @u1(%cst, %arg0) {qasm.gate} : (f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}

func @main() {
  %0 = qasm.allocate
  call @x(%0) {qasm.gate} : (!qasm.qubit) -> ()
  call @y(%0) {qasm.gate}: (!qasm.qubit) -> ()
  call @z(%0) {qasm.gate}: (!qasm.qubit) -> ()
  return
}
