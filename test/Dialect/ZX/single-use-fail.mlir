// RUN: quantum-opt %s | quantum-opt
// RUN: quantum-opt --zx-check-single-use %s 2>&1 | FileCheck %s

func @foo() {
  // CHECK: error: ZX Wire declared here is used multiple times.
  %0 = zx.source
  zx.sink %0
  zx.sink %0
  return
}

func @bar() {
  // CHECK: error: ZX Wire declared here is not used.
  %0 = zx.source
  return
}
