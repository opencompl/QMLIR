// RUN: quantum-opt --show-dialects | FileCheck --check-prefix="CHECK-DIALECT" %s
// RUN: quantum-opt --help-list | FileCheck --check-prefix="CHECK-OPTION" %s

// CHECK-DIALECT: scf
// CHECK-DIALECT: std
// CHECK-DIALECT: zx
// CHECK-OPTION: --apply-zx-rewrites
// CHECK-OPTION: --zx-check-single-use
