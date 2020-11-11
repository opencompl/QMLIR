module {
  func @main() {
    %q0 = quantum.allocate() : !quantum.qubit<3>
    %q1 = quantum.pauliX %q0 : !quantum.qubit<3>
    %r = quantum.measure %q1 : !quantum.qubit<3> -> memref<3xi1>
    return
  }
}

