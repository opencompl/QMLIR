// RUN: quantum-opt --qssa-apply-rewrites %s

func @swap() {
  %a0 = qssa.alloc {qubit = "a"} : !qssa.qubit<1>
  %b0 = qssa.alloc {qubit = "b"} : !qssa.qubit<1>
  %a1, %b1 = qssa.CNOT %a0, %b0
  %b2, %a2 = qssa.CNOT %b1, %a1
  %a3, %b3 = qssa.CNOT %a2, %b2
  qssa.sink %a3 {qubit = "a"} : !qssa.qubit<1>
  qssa.sink %b3 {qubit = "b"} : !qssa.qubit<1>
  return
}
func @alternateCNOT() {
  %a0 = qssa.alloc {qubit = "a"} : !qssa.qubit<1>
  %b0 = qssa.alloc {qubit = "b"} : !qssa.qubit<1>
  %a1, %b1 = qssa.CNOT %a0, %b0
  %b2, %a2 = qssa.CNOT %b1, %a1
  qssa.sink %a2 {qubit = "a"} : !qssa.qubit<1>
  qssa.sink %b2 {qubit = "b"} : !qssa.qubit<1>
  return
}
