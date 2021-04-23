// RUN: quantum-opt %s --qssa-prepare-for-zx --convert-qssa-to-zx

func @cnot() {
  %a = qssa.alloc {name="a"} : !qssa.qubit<1>
  %b = qssa.alloc {name="b"} : !qssa.qubit<1>
  %a1, %b1 = qssa.CNOT %a, %b
  qssa.sink %a1 {name="a"} : !qssa.qubit<1>
  qssa.sink %b1 {name="b"} : !qssa.qubit<1>
  return
}
