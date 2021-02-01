// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --apply-zxg-rewrites %s | quantum-opt | FileCheck %s

// CHECK: func @fusion() {
// CHECK-NEXT: return
// CHECK-NEXT: }
func @fusion() {
  %zero = constant 0.0 : f32
  %1 = zxg.Z %zero : f32
  %2 = zxg.Z %zero : f32
  zxg.wire %1 %2
  return
}

func @fusionChain() {
  %zero = constant 0.0 : f32
  // CHECK: %[[A:.*]] = zxg.terminal
  %0 = zxg.terminal
  // CHECK: %[[B:.*]] = zxg.Z
  %1 = zxg.Z %zero : f32
  %2 = zxg.Z %zero : f32
  %3 = zxg.Z %zero : f32
  // CHECK: %[[C:.*]] = zxg.terminal
  %4 = zxg.terminal

  // CHECK: zxg.wire %[[A]] %[[B]]
  // CHECK: zxg.wire %[[B]] %[[C]]
  zxg.wire %0 %1
  zxg.wire %1 %2
  zxg.wire %2 %3
  zxg.wire %3 %4

  return
}

