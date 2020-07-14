// RUN: quantum-opt %s | quantum-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = quantum.foo %{{.*}} : i32
        %res = quantum.foo %0 : i32
        return
    }
}
