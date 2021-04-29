// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt -apply-zx-rewrites %s
// no-run: quantum-opt -apply-zx-rewrites %s | FileCheck %s

func @varfoo(%u : f32, %v : f32, %w: f32) {
  %a = zx.source
  %b = zx.source
  %c = zx.source
  %d = zx.source
  %e = zx.source

  %x, %y, %m1 = zx.Z(%u : f32) %a, %b
                  : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire, !zx.wire)
  %z, %m2 = zx.Z(%v : f32) %m1, %c, %d
                  : (!zx.wire, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %p, %q = zx.Z(%w : f32) %m2, %e : (!zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  zx.sink %x
  zx.sink %y
  zx.sink %z
  zx.sink %p
  zx.sink %q

  return
}

