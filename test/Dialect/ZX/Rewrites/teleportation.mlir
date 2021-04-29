// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zx-check-single-use
// RUN: quantum-opt --apply-zx-rewrites %s | quantum-opt

func @CNOT(%a0 : !zx.wire, %b0 : !zx.wire) -> (!zx.wire, !zx.wire) {
  %zero = constant 0.0 : f32
  %a1, %m = zx.Z(%zero : f32) %a0 : (!zx.wire) -> (!zx.wire, !zx.wire)
  %b1 = zx.X(%zero : f32) %m, %b0 : (!zx.wire, !zx.wire) -> (!zx.wire)
  return %a1, %b1 : !zx.wire, !zx.wire
}

func @teleport() {
  %zero = constant 0.0 : f32

  // particle to be teleported
  %a0 = zx.source
  // (shared) entangled bell pair
  %b0, %c0 = zx.X(%zero : f32) : () -> (!zx.wire, !zx.wire)

  // Alice has : %a0, %b0.
  // Bob has   : %c0

  // CNOT a, b
  %a1, %m0 = zx.Z(%zero : f32) %a0 : (!zx.wire) -> (!zx.wire, !zx.wire)
  %b1 = zx.X(%zero : f32) %m0, %b0 : (!zx.wire, !zx.wire) -> (!zx.wire)

  %a2 = zx.H %a1

  // CNOT b, c
  %b2, %m1 = zx.Z(%zero : f32) %b1 : (!zx.wire) -> (!zx.wire, !zx.wire)
  %c1 = zx.X(%zero : f32) %m1, %c0 : (!zx.wire, !zx.wire) -> (!zx.wire)

  %c2 = zx.H %c1

  // CNOT a, c
  %a3, %m2 = zx.Z(%zero : f32) %a2 : (!zx.wire) -> (!zx.wire, !zx.wire)
  %c3 = zx.X(%zero : f32) %c2, %m2 : (!zx.wire, !zx.wire) -> (!zx.wire)

  %c4 = zx.H %c3

  zx.sink %a3
  zx.sink %b2
  zx.sink %c4

  return
}
