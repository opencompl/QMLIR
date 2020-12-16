module {

func @allocate_op() {
  %q = qasm.allocate : !qasm.qubit<10>
  %a = qasm.allocate : !qasm.qubit<1>
  %b = qasm.allocate : !qasm.qubit<1>
  qasm.CX %a, %b : !qasm.qubit<1>
  %t = constant 0.0 : f32
  qasm.U (%t : f32, %t : f32, %t : f32) %a : !qasm.qubit<1>
  qasm.gphase(%t : f32)
  return
}

}
