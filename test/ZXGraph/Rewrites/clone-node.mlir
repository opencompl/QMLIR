// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-canonicalize-blocks --zxg-apply-rewrites %s | quantum-opt

func @cloneX(%alpha : f32) {
  %zero = constant 0.0 : f32
  %0 = zxg.X %zero : f32
  %1 = zxg.Z %alpha : f32
  %x = zxg.terminal
  %y = zxg.terminal
  %z = zxg.terminal
  zxg.wire %0 %1
  zxg.wire %1 %x
  zxg.wire %1 %y
  zxg.wire %1 %z
  return
}
