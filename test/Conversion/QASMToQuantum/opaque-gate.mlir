// RUN: quantum-opt %s
// RUN: quantum-opt --qasm-make-gates-opaque=gates=s,sdg,t,tdg,rx,ry,rz --convert-qasm-to-qssa %s | quantum-opt

func private @u3(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "u3"} {
  qasm.U(%arg0 : f64, %arg1 : f64, %arg2 : f64) %arg3
  return {qasm.gate_end}
}
func private @u2(%arg0: f64, %arg1: f64, %arg2: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "u2"} {
  %cst = constant 1.5707963267948966 : f64
  qasm.U(%cst : f64, %arg0 : f64, %arg1 : f64) %arg2
  return {qasm.gate_end}
}
func private @u1(%arg0: f64, %arg1: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "u1"} {
  %cst = constant 0.000000e+00 : f64
  qasm.U(%cst : f64, %cst : f64, %arg0 : f64) %arg1
  return {qasm.gate_end}
}
func private @cx(%arg0: !qasm.qubit, %arg1: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "cx"} {
  qasm.CX %arg0, %arg1
  return {qasm.gate_end}
}
func private @h(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "h"} {
  %cst = constant 3.1415926535897931 : f64
  %cst_0 = constant 0.000000e+00 : f64
  call @u2(%cst_0, %cst, %arg0) {qasm.gate} : (f64, f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @s(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "s"} {
  %cst = constant 1.5707963267948966 : f64
  call @u1(%cst, %arg0) {qasm.gate} : (f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @sdg(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "sdg"} {
  %cst = constant -1.5707963267948966 : f64
  call @u1(%cst, %arg0) {qasm.gate} : (f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @t(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "t"} {
  %cst = constant 0.78539816339744828 : f64
  call @u1(%cst, %arg0) {qasm.gate} : (f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @tdg(%arg0: !qasm.qubit) attributes {qasm.gate, qasm.stdgate = "tdg"} {
  %cst = constant -0.78539816339744828 : f64
  call @u1(%cst, %arg0) {qasm.gate} : (f64, !qasm.qubit) -> ()
  return {qasm.gate_end}
}
func private @rx (%0 : f64, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate="rx"}
{
  %2 = qasm.pi : f64
  %3 = std.negf %2 : f64
  %4 = std.constant 2.000000e+00 : f64
  %5 = std.divf %3, %4 : f64
  %6 = qasm.pi : f64
  %7 = std.constant 2.000000e+00 : f64
  %8 = std.divf %6, %7 : f64
  std.call @u3(%0, %5, %8, %1) {qasm.gate} : (f64, f64, f64, !qasm.qubit) -> ()
  std.return {qasm.gate_end}
}
func private @ry (%0 : f64, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate="ry"}
{
  %zero = std.constant 0.000000e+00 : f64
  std.call @u3(%0, %zero, %zero, %1) {qasm.gate} : (f64, f64, f64, !qasm.qubit) -> ()
  std.return {qasm.gate_end}
}
func private @rz (%0 : f64, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate="rz"}
{
  std.call @u1(%0, %1) {qasm.gate} : (f64, !qasm.qubit) -> ()
  std.return {qasm.gate_end}
}

func @qasm_main () -> (i1, i1, i1) {
  %cst = constant 1.0 : f64
  %0 = qasm.allocate
  %1 = qasm.allocate
  %2 = qasm.allocate

  call @s(%0) {qasm.gate} : (!qasm.qubit) -> ()
  call @sdg(%0) {qasm.gate} : (!qasm.qubit) -> ()
  call @t(%1) {qasm.gate} : (!qasm.qubit) -> ()
  call @tdg(%1) {qasm.gate} : (!qasm.qubit) -> ()
  call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
  call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
  call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
  call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
  call @rx(%cst, %2) {qasm.gate} : (f64, !qasm.qubit) -> ()
  call @ry(%cst, %2) {qasm.gate} : (f64, !qasm.qubit) -> ()
  call @rz(%cst, %2) {qasm.gate} : (f64, !qasm.qubit) -> ()

  %r0 = qasm.measure %0
  %r1 = qasm.measure %1
  %r2 = qasm.measure %2
  return %r0, %r1, %r2 : i1, i1, i1
}
