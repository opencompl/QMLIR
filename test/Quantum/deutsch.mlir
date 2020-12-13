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
func private @oracle(%x : !quantum.qubit<?>, %y : !quantum.qubit<1>)
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

// return false for constant, true for balanced
func @deutsch_josza(%n : index) -> i1 { // %n : number of input bits
  %x0 = quantum.allocate(%n) : !quantum.qubit<?>
  %x1 = quantum.H %x0 : !quantum.qubit<?>
  %x2 = call @phase_flip_oracle(%x1) : (!quantum.qubit<?>) -> !quantum.qubit<?>
  %x3 = quantum.H %x2 : !quantum.qubit<?>
  %res = quantum.measure %x3 : !quantum.qubit<?> -> memref<?xi1>

  // compute bitwise-OR of all the bits in %res
  %false = constant 0 : i1
  %0 = constant 0 : index
  %1 = constant 1 : index
  %lst = subi %n, %1 : index

  %ans = scf.for %i = %0 to %lst step %1
    iter_args(%out = %false) -> i1 {
    %v = load %res[%i] : memref<?xi1>
    %cur = or %out, %v : i1
    scf.yield %cur : i1
  }

  return %ans : i1
}
