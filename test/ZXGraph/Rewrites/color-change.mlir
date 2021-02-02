// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-apply-rewrites %s | quantum-opt | FileCheck %s

func @ZtoX() {
  %zero = constant 0.0 : f32
  %0 = zxg.Z %zero : f32
  %1 = zxg.H
  %2 = zxg.H
  %3 = zxg.H
  %4 = zxg.terminal
  %5 = zxg.terminal
  %6 = zxg.terminal

  zxg.wire %0 %1
  zxg.wire %0 %2
  zxg.wire %0 %3
  zxg.wire %1 %4
  zxg.wire %2 %5
  zxg.wire %3 %6

  return
}
