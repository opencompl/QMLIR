// RUN: quantum-opt %s | quantum-opt 

module {
  func @main() {
    %qs = quantum.allocate : !quantum.qubit<2>

    return
  }
}