// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --apply-zxg-rewrites %s | quantum-opt

func @teleport() {
  %zero = constant 0.0 : f32

  // Input (A)
  %psi = zxg.terminal
  // Input bell pair (A, B)
  %eA = zxg.Z %zero : f32
  %eB = zxg.Z %zero : f32
  zxg.wire %eA %eB

  // Output (A, A, B)
  %eA_fin = zxg.terminal
  %eB_fin = zxg.terminal
  %psi_fin = zxg.terminal

  %l0 = zxg.Z %zero : f32
  %r0 = zxg.X %zero : f32
  zxg.wire %psi %l0
  zxg.wire %eA %r0
  zxg.wire %l0 %r0

  %l1 = zxg.Z %zero : f32
  %r1 = zxg.X %zero : f32
  zxg.wire %r0 %l1
  zxg.wire %eB_fin %l1
  zxg.wire %eB %r1
  zxg.wire %l1 %r1

  %h0 = zxg.H
  zxg.wire %l0 %h0

  %l2 = zxg.Z %zero : f32
  %r2 = zxg.X %zero : f32
  %h1 = zxg.H
  zxg.wire %h0 %l2
  zxg.wire %l2 %eA_fin
  zxg.wire %l2 %h1
  zxg.wire %h1 %r2
  zxg.wire %r1 %r2
  zxg.wire %r2 %psi_fin

  return
}
