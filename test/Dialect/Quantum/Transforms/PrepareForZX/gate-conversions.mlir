// RUN: quantum-opt %s --qssa-prepare-for-zx | FileCheck %s

func @convertT() {
  // CHECK: %[[cst:.*]] = constant 0.78539816339744828 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.T %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertTdg() {
  // CHECK: %[[cst:.*]] = constant -0.78539816339744828 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.Tdg %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertS() {
  // CHECK: %[[cst:.*]] = constant 1.5707963267948966 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.S %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertSdg() {
  // CHECK: %[[cst:.*]] = constant -1.5707963267948966 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.Sdg %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertX() {
  // CHECK: %[[cst:.*]] = constant 3.1415926535897931 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rx(%[[cst]] : f64)
  %1 = qssa.X %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertZ() {
  // CHECK: %[[cst:.*]] = constant 3.1415926535897931 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.Z %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertH() {
  // CHECK: %[[cst:.*]] = constant 1.5707963267948966 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[cst]] : f64)
  // CHECK: qssa.Rx(%[[cst]] : f64)
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.H %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertY() {
  // CHECK: %[[cst:.*]] = constant 1.5707963267948966 : f64
  // CHECK: %[[neg:.*]] = constant -1.5707963267948966 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[neg]] : f64)
  // CHECK: qssa.Rx(%[[cst]] : f64)
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.Y %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

// CHECK: func @convertRy(%[[alpha:.*]]: f64)
func @convertRy(%alpha : f64) {
  // CHECK: %[[cst:.*]] = constant 1.5707963267948966 : f64
  // CHECK: %[[neg:.*]] = constant -1.5707963267948966 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  // CHECK: qssa.Rz(%[[neg]] : f64)
  // CHECK: qssa.Rx(%[[alpha]] : f64)
  // CHECK: qssa.Rz(%[[cst]] : f64)
  %1 = qssa.Ry(%alpha : f64) %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}

func @convertU(%theta : f64, %phi : f64, %lambda : f64) {
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.U(%theta : f64, %phi : f64, %lambda : f64) %0 : !qssa.qubit<1>
  qssa.sink %1 : !qssa.qubit<1>
  return
}
