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

  // CHECK: %[[C:.*]] = constant [[_:.*]]: f32

  // CHECK: %[[INP:.*]]:3 = zx.source
  %a, %b, %c = zx.source() : () -> (!zx.wire, !zx.wire, !zx.wire)
  %m, %x = zx.Z(%u, %a, %b) : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %y, %z = zx.Z(%v, %c, %m) : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  // CHECK: %[[OUT:.*]]:3 = zx.Z(%[[C]], %[[INP]]#0, %[[INP]]#1, %[[INP]]#2)
  // CHECK: zx.sink(%[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2)
  zx.sink(%x, %y, %z) : (!zx.wire, !zx.wire, !zx.wire) -> ()

  return
}

// CHECK: func @varfoo(%[[U:.*]]: f32, %[[V:.*]]: f32)
func @varfoo(%u : f32, %v : f32) {

  // CHECK: %[[INP:.*]]:3 = zx.source
  %a, %b, %c = zx.source() : () -> (!zx.wire, !zx.wire, !zx.wire)

  // CHECK: %[[W:.*]] = addf %[[U]], %[[V]] : f32
  // CHECK: %[[OUT:.*]]:3 = zx.Z(%[[W]], %[[INP]]#0, %[[INP]]#1, %[[INP]]#2)
  %m, %x = zx.Z(%u, %a, %b) : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %y, %z = zx.Z(%v, %c, %m) : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  // CHECK: zx.sink(%[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2)
  zx.sink(%x, %y, %z) : (!zx.wire, !zx.wire, !zx.wire) -> ()

  return
}

