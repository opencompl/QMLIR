// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zxg-canonicalize-blocks --zxg-apply-rewrites %s | quantum-opt

func @had_chain_2() {
  %0 = zxg.terminal
  %1 = zxg.H
  %2 = zxg.H
  %3 = zxg.terminal
  zxg.wire %0 %1
  zxg.wire %1 %2
  zxg.wire %2 %3
  return
}

func @had_chain_3() {
  %0 = zxg.terminal
  %1 = zxg.H
  %2 = zxg.H
  %3 = zxg.H
  %4 = zxg.terminal
  zxg.wire %0 %1
  zxg.wire %1 %2
  zxg.wire %2 %3
  zxg.wire %3 %4
  return
}
