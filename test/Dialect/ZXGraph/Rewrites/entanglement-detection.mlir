// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-canonicalize-blocks --zxg-apply-rewrites %s | quantum-opt

func @unitary() -> (!zxg.node, !zxg.node, !zxg.node, !zxg.node) {
  %zero = constant 0.0 : f32
  %half = constant 0.5 : f32

  %0 = zxg.X %zero : f32
  %1 = zxg.X %zero : f32
  %2 = zxg.X %zero : f32
  %3 = zxg.X %zero : f32

  %4 = zxg.H
  zxg.wire %2 %4

  %5 = zxg.Z %half : f32
  zxg.wire %4 %5

  %6 = zxg.Z %zero : f32
  %7 = zxg.X %zero : f32
  zxg.wire %5 %6
  zxg.wire %3 %7
  zxg.wire %6 %7

  %8 = zxg.Z %zero : f32
  %9 = zxg.X %zero : f32
  zxg.wire %6 %9
  zxg.wire %0 %8
  zxg.wire %8 %9

  %10 = zxg.Z %zero : f32
  %11 = zxg.X %zero : f32
  zxg.wire %1 %10
  zxg.wire %3 %11
  zxg.wire %10 %11

  %12 = zxg.X %zero : f32
  %13 = zxg.Z %zero : f32
  zxg.wire %10 %12
  zxg.wire %11 %13
  zxg.wire %12 %13

  %14 = zxg.X %zero : f32
  %15 = zxg.Z %zero : f32
  zxg.wire %8 %14
  zxg.wire %12 %15
  zxg.wire %14 %15

  %16 = zxg.H
  zxg.wire %15 %16

  %17 = zxg.X %zero : f32
  %18 = zxg.Z %zero : f32
  zxg.wire %16 %17
  zxg.wire %9 %18
  zxg.wire %17 %18

  %19 = zxg.X %zero : f32
  %20 = zxg.Z %zero : f32
  zxg.wire %18 %19
  zxg.wire %13 %20
  zxg.wire %19 %20

  %21 = zxg.X %zero : f32
  %22 = zxg.Z %zero : f32
  zxg.wire %17 %21
  zxg.wire %20 %22
  zxg.wire %21 %22

  %23 = zxg.Z %zero : f32
  %24 = zxg.X %zero : f32
  zxg.wire %14 %23
  zxg.wire %19 %24
  zxg.wire %23 %24

  %25 = zxg.Z %half : f32
  zxg.wire %21 %25

  %26 = zxg.Z %zero : f32
  %27 = zxg.X %zero : f32
  zxg.wire %23 %26
  zxg.wire %25 %27
  zxg.wire %26 %27

  %28 = zxg.terminal
  %29 = zxg.terminal
  %30 = zxg.terminal
  %31 = zxg.terminal
  zxg.wire %26 %28
  zxg.wire %27 %29
  zxg.wire %24 %30
  zxg.wire %22 %31

  return %28, %29, %30, %31 : !zxg.node, !zxg.node, !zxg.node, !zxg.node
}
