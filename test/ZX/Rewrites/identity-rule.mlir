// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --apply-zx-rewrites %s | quantum-opt
// RUN: quantum-opt --zx-check-single-use

func @id() {
  %alpha = constant 0.0 : f32
  %0 = zx.source
  %1 = zx.Z(%alpha, %0) : (f32, !zx.wire) -> (!zx.wire)
  zx.sink %1
  return
}

func @id_with_scf() {
  %alpha = constant 0.0 : f32
  %zero = constant 0.0 : f32
  %0 = zx.source
  %cond = cmpf "ueq", %alpha, %zero : f32
  %1 = scf.if %cond -> !zx.wire {
    scf.yield %0 : !zx.wire
  } else {
    %res = zx.Z(%alpha, %0) : (f32, !zx.wire) -> (!zx.wire)
    scf.yield %res : !zx.wire
  }
  zx.sink %1
  return
}

func @id_with_fusion() {
  %zero = constant 0.0 : f32
  %pi = constant 3.141592653589793 : f32

  %0 = zx.source
  %1 = zx.Z(%pi, %0) : (f32, !zx.wire) -> (!zx.wire)
  %2 = zx.Z(%pi, %1) : (f32, !zx.wire) -> (!zx.wire)
  zx.sink %2

  return
}
