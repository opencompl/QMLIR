//===- deutsch.mlir - The Deutsch-Josza Algorithm -------------------------===//
// Given a quantum oracle for a function `f : {0, 1}ⁿ → {0, 1}`
// and that `f` is either balanced or constant, detect which one it is.
// 
// Definitions:
// - Balanced function: || {x | f(x) = 0} || = || {x | f(x) = 1} ||
// - Constant function: ∃ k, ∀ x, f(x) = k
// - Oracle: A function evaluation oracle.
//     In our case, the quantum oracle takes in `n` input qubits, and 
//     one output qubit, and transforms as follows:
//     U |x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
//===----------------------------------------------------------------------===//

// RUN: quantum-opt %s | quantum-opt 

// implements U|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
func @oracle(%x : !quantum.qubit<?>, %y : !quantum.qubit<1>)
  -> (!quantum.qubit<?>, !quantum.qubit<1>)

// implements U|x⟩ = (-1)^{f(x)} |x⟩
func @phase_flip_oracle(%x : !quantum.qubit<?>)
  -> !quantum.qubit<?> {
  %y0 = quantum.allocate() : !quantum.qubit<1>
  %y1 = quantum.pauliX %y0 : !quantum.qubit<1>
  %y2 = quantum.H %y1 : !quantum.qubit<1>
  %x1, %y3 = call @oracle(%x, %y2)
    : (!quantum.qubit<?>, !quantum.qubit<1>) -> (!quantum.qubit<?>, !quantum.qubit<1>)

  %0 = quantum.measure %y3 : !quantum.qubit<1> -> memref<1xi1>

  return %x1: !quantum.qubit<?>
}

func @applyH(%qs : !quantum.qubit<?>) -> !quantum.qubit<?> {
  %1 = constant 1 : index
  %qs1, %n = quantum.dim %qs : !quantum.qubit<?>
  %nminus1 = subi %n, %1 : index

  %qf = scf.for %i = %1 to %n step %1
    iter_args(%q0 = %qs1) -> !quantum.qubit<?> {
    %qh, %qr = quantum.split %q0[%nminus1] : !quantum.qubit<?> -> (!quantum.qubit<1>, !quantum.qubit<?>)
    %qh1 = quantum.H %qh : !quantum.qubit<1>
    %q1 = quantum.concat %qr, %qh1 : (!quantum.qubit<?>, !quantum.qubit<1>) -> !quantum.qubit<?>
    scf.yield %q1 : !quantum.qubit<?>
  }

  return %qf : !quantum.qubit<?>
}

// return false for constant, true for balanced
func @deutsch_josza() -> i1 {
  %x0 = quantum.allocate() : !quantum.qubit<10>
  %x1 = quantum.cast %x0 : !quantum.qubit<10> to !quantum.qubit<?>
  %x2 = call @applyH(%x1) : (!quantum.qubit<?>) -> !quantum.qubit<?>
  %x3 = call @phase_flip_oracle(%x2) : (!quantum.qubit<?>) -> !quantum.qubit<?>
  %x4 = call @applyH(%x3) : (!quantum.qubit<?>) -> !quantum.qubit<?>
  %x5 = quantum.cast %x4 : !quantum.qubit<?> to !quantum.qubit<10>
  %res = quantum.measure %x5 : !quantum.qubit<10> -> memref<10xi1>

  %false = constant 0 : i1
  %0 = constant 0 : index
  %1 = constant 1 : index
  %n = constant 10 : index
  %lst = subi %n, %1 : index

  %ans = scf.for %i = %0 to %lst step %1
    iter_args(%out = %false) -> i1 {
    %v = load %res[%i] : memref<10xi1>
    %cur = or %out, %v : i1
    scf.yield %cur : i1
  }

  return %ans : i1
}
