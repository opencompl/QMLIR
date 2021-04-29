// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt -apply-zx-rewrites %s
// RUN: quantum-opt -apply-zx-rewrites %s | FileCheck %s

// ZX Graph:
//  a_____        ___________________x
//        \      /
//         \    /
//  b------[Z(u)]---m---[Z(v)]-------y
//                       /  \
//  c___________________/    \_______z

// After rewrite:
//  a____________       _____________x
//               \     /
//                \   /
//  b------------[Z(u+v)]------------y
//                /   \
//  c____________/     \_____________z

func @constfoo() {
  %u = constant 1.0 : f32
  %v = constant 2.0 : f32

  // CHECK: %[[INP1:.*]] = zx.source
  // CHECK: %[[INP2:.*]] = zx.source
  // CHECK: %[[INP3:.*]] = zx.source
  %a = zx.source
  %b = zx.source
  %c = zx.source
  %m, %x = zx.Z(%u : f32) %a, %b : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %y, %z = zx.Z(%v : f32) %c, %m : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  // CHECK: %[[OUT:.*]]:3 = zx.Z(%[[C:.*]] : f32) %[[INP1]], %[[INP2]], %[[INP3]]
  // CHECK: zx.sink %[[OUT]]#0
  // CHECK: zx.sink %[[OUT]]#1
  // CHECK: zx.sink %[[OUT]]#2
  zx.sink %x
  zx.sink %y
  zx.sink %z

  return
}

// CHECK: func @varfoo(%[[U:.*]]: f32, %[[V:.*]]: f32)
func @varfoo(%u : f32, %v : f32) {
  // CHECK: %[[INP1:.*]] = zx.source
  // CHECK: %[[INP2:.*]] = zx.source
  // CHECK: %[[INP3:.*]] = zx.source
  %a = zx.source
  %b = zx.source
  %c = zx.source

  // CHECK: %[[OUT:.*]]:3 = zx.Z(%[[W:.*]] : f32) %[[INP1]], %[[INP2]], %[[INP3]]
  %m, %x = zx.Z(%u : f32) %a, %b : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %y, %z = zx.Z(%v : f32) %c, %m : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  // CHECK: zx.sink %[[OUT]]#0
  // CHECK: zx.sink %[[OUT]]#1
  // CHECK: zx.sink %[[OUT]]#2
  zx.sink %x
  zx.sink %y
  zx.sink %z

  return
}

