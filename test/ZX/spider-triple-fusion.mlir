// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt -apply-zx-rewrites %s
// no-run: quantum-opt -apply-zx-rewrites %s | FileCheck %s

func @varfoo(%u : f32, %v : f32, %w: f32) {
  %a, %b, %c, %d, %e = zx.source()
                         : () -> (!zx.wire, !zx.wire, !zx.wire, !zx.wire, !zx.wire)

  %x, %y, %m1 = zx.Z(%u, %a, %b)
                  : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire, !zx.wire)
  %z, %m2 = zx.Z(%v, %m1, %c, %d)
                  : (f32, !zx.wire, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)
  %p, %q = zx.Z(%w, %m2, %e) : (f32, !zx.wire, !zx.wire) -> (!zx.wire, !zx.wire)

  zx.sink(%x, %y, %z) : (!zx.wire, !zx.wire, !zx.wire) -> ()
  zx.sink(%p, %q) : (!zx.wire, !zx.wire) -> ()

  return
}

