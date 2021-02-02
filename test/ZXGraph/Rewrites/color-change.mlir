// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-canonicalize-blocks --zxg-apply-rewrites %s | quantum-opt | FileCheck %s

func @ZtoX() {
  %zero = constant 0.0 : f32
  // CHECK: %[[z:.*]] = zxg.X
  %0 = zxg.Z %zero : f32
  %1 = zxg.H
  %2 = zxg.H
  %3 = zxg.H
  // CHECK: %[[a:.*]] = zxg.terminal
  %4 = zxg.terminal
  // CHECK: %[[b:.*]] = zxg.terminal
  %5 = zxg.terminal
  // CHECK: %[[c:.*]] = zxg.terminal
  %6 = zxg.terminal

  // CHECK: zxg.wire %[[z]] %[[c]]
  zxg.wire %0 %1
  zxg.wire %1 %4
  // CHECK: zxg.wire %[[z]] %[[b]]
  zxg.wire %0 %2
  zxg.wire %2 %5
  // CHECK: zxg.wire %[[z]] %[[a]]
  zxg.wire %0 %3
  zxg.wire %3 %6

  return
}
