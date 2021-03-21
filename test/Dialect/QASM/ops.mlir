// RUN: quantum-opt %s | quantum-opt

module {
  func @qasm_main() {
    %q = qasm.allocate
    %a = qasm.allocate
    %b = qasm.allocate
    qasm.CX %a, %b
    %t = constant 0.0 : f32
    qasm.U (%t : f32, %t : f32, %t : f32) %a
    qasm.gphase(%t : f32)
    qasm.reset %q
    return
  }
}
