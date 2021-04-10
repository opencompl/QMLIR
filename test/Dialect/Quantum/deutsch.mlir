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
func private @oracle(%x : !qssa.qubit<?>, %y : !qssa.qubit<1>)
  -> (!qssa.qubit<?>, !qssa.qubit<1>)

// implements U|x⟩ = (-1)^{f(x)} |x⟩
func @phase_flip_oracle(%x : !qssa.qubit<?>)
  -> !qssa.qubit<?> {
  %y0 = qssa.alloc() : !qssa.qubit<1>
  %y1 = qssa.X %y0 : !qssa.qubit<1>
  %y2 = qssa.H %y1 : !qssa.qubit<1>
  %x1, %y3 = call @oracle(%x, %y2)
    : (!qssa.qubit<?>, !qssa.qubit<1>) -> (!qssa.qubit<?>, !qssa.qubit<1>)

  // qssa.measure %y3 -> %res : !qssa.qubit<1> -> memref<1xi1>

  return %x1: !qssa.qubit<?>
}

// return false for constant, true for balanced
func @deutsch_josza(%n : index) -> i1 { // %n : number of input bits
  %x0 = qssa.alloc(%n) : !qssa.qubit<?>
  %x1 = qssa.H %x0 : !qssa.qubit<?>
  %x2 = call @phase_flip_oracle(%x1) : (!qssa.qubit<?>) -> !qssa.qubit<?>
  %x3 = qssa.H %x2 : !qssa.qubit<?>
  %res, %x4 = qssa.measure %x3 : !qssa.qubit<?> -> tensor<?xi1>

  // compute bitwise-OR of all the bits in %res
  %false = constant 0 : i1
  %0 = constant 0 : index
  %1 = constant 1 : index
  %lst = subi %n, %1 : index

  %ans = scf.for %i = %0 to %lst step %1
    iter_args(%out = %false) -> i1 {
    %v = tensor.extract %res[%i] : tensor<?xi1>
    %cur = or %out, %v : i1
    scf.yield %cur : i1
  }

  return %ans : i1
}
