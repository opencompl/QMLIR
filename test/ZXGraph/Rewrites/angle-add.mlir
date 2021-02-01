// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --apply-zxg-rewrites %s | quantum-opt | FileCheck %s

func @moduloCheck() {
  %one = constant 1.5 : f32
  %0 = zxg.terminal
  %1 = zxg.Z %one : f32
  %2 = zxg.Z %one : f32
  %3 = zxg.terminal

  zxg.wire %0 %1
  zxg.wire %1 %2
  zxg.wire %2 %3

  return
}
