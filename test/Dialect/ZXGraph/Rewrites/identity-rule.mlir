// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-apply-rewrites %s | quantum-opt | FileCheck %s

func @identity() {
  %zero = constant 0.0 : f32
  // CHECK: %[[A:.*]] = zxg.terminal
  %0 = zxg.terminal
  %1 = zxg.Z %zero : f32
  // CHECK: %[[B:.*]] = zxg.terminal
  %2 = zxg.terminal

  zxg.wire %0 %1
  zxg.wire %1 %2

  return
}
