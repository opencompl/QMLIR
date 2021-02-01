// RUN: quantum-opt %s | quantum-opt

// (A) wire to merge
func @foo() {
  %zero = constant 0.0 : f32
  %0 = zxg.Z %zero : f32
  %1 = zxg.X %zero : f32
  %2 = zxg.H
  zxg.wire %0 %1
  return
}

func @canon() {
  %zero = constant 0.0 : f32
  %0 = zxg.Z %zero : f32
  zxg.wire %0 %0
  return
}

// (B) wire to declare
// func @bar() {
//   %zero = constant 0.0 : f32
//   %0, %1 = zxg.wire
//   %2, %3 = zxg.wire
//   zxg.Z(%zero, %0, %2)
//   zxg.X(%zero, %1)
//   zxg.sink %3
// }

// func @CNOT(%0 : node, %1 : node
//            %a : node, %b : node) {
//   %wl, %wr = wire
//
//   X(%0, %a, %wl)
//   Z(%1, %b, %wr)
//   H(%0, %1)
//   /// equiv
//   // X(%0, %a, %wr)
//   // Z(%1, %b, %wl)
//
//   // or annotate at the end (?)
//   // wire %wl %wr
//
//   return
// }
//
// func CNOTv2() {
//   %p, %q, %r = X
//   // wire %0 %p
//   // wire %a %q
//   %p2, %q2, %r2 = Z
//   // wire %1 %p2
//   // wire %b %q2
//   wire %r  %r2
//   return %p %p2 %q %q2
// }
//
// // decision: wiring at caller or callee (?)
//
// %... qcall (X, %...)
