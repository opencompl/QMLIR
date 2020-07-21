// RUN: quantum-opt %s | quantum-opt 

module {
  func @main() {
    %X = quantum.gate_prim : !quantum.gate<1>

    %q1 = quantum.allocate : !quantum.qubit<2>
    %q2 = quantum.allocate : !quantum.qubit<3>
    
    %q3 = quantum.concat (%q1:!quantum.qubit<2>), (%q2:!quantum.qubit<3>) : !quantum.qubit<5>
    %q41, %q42 = quantum.split (%q3:!quantum.qubit<5>) : (!quantum.qubit<4>, !quantum.qubit<1>)
    
    %q5 = quantum.transform (%q42: !quantum.qubit<1>), (%X: !quantum.gate<1>) : !quantum.qubit<1>
    %res = quantum.measure (%q5:!quantum.qubit<1>) : i64
    
    return
  }
}
