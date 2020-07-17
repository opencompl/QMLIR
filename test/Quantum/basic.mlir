// RUN: quantum-opt %s | quantum-opt 

module {
  func @main() {
    %q1 = quantum.allocate : !quantum.qubit<2>
    %q2 = quantum.allocate : !quantum.qubit<3>
    %q3 = quantum.concat (%q1:!quantum.qubit<2>), (%q2:!quantum.qubit<3>) : !quantum.qubit<5>
    %q4 = quantum.concat (%q1:!quantum.qubit<2>), (%q2:!quantum.qubit<3>) : !quantum.qubit<5>
    %res = quantum.measure (%q4:!quantum.qubit<5>) : i64
    return
  }
}
