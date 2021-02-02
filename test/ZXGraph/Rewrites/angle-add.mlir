// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-canonicalize-blocks --zxg-apply-rewrites %s | quantum-opt | FileCheck %s

func @moduloCheck() {
  // CHECK: %[[cst:.*]] = constant 1.000
  %one = constant 1.5 : f32
  // CHECK: %[[l:.*]] = zxg.terminal
  %0 = zxg.terminal
  // CHECK: %[[m:.*]] = zxg.Z %cst : f32
  %1 = zxg.Z %one : f32
  %2 = zxg.Z %one : f32
  // CHECK: %[[r:.*]] = zxg.terminal
  %3 = zxg.terminal

  // CHECK: zxg.wire %[[l]] %[[m]]
  // CHECK: zxg.wire %[[m]] %[[r]]
  zxg.wire %0 %1
  zxg.wire %1 %2
  zxg.wire %2 %3

  return
}
