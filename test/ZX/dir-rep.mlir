// RUN: quantum-opt %s | quantum-opt

func @CNOT(%0: !zx.wire, %1 : !zx.wire) -> (!zx.wire, !zx.wire) {
  // inputs %0 %1
  %alpha = constant 0.0 : f32
  %a, %m = zx.Z[%alpha : f32] (%0) : (!zx.wire, !zx.wire)
  %b = zx.X [%alpha : f32] (%m, %1): (!zx.wire)
  // outputs %a %b

  return %a, %b : !zx.wire, !zx.wire
}

// func @CNOT(%0: !zx.wire, %1 : !zx.wire) {
//   // inputs %0 %1
//   %b, %n = X(%1)
//   %a = Z(%n, %0)
//   // outputs %a %b
//
//   return %a, %b
// }
