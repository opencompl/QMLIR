// RUN: quantum-opt %s --qssa-prepare-for-zx --convert-qssa-to-zx

func @alloc() {
  %0 = qssa.alloc : !qssa.qubit<1>
  %res, %1 = qssa.measure_one %0
  return
}

// func @allocMultiqubit() {
//   %0 = qssa.alloc : !qssa.qubit<2>
//   %res, %1 = qssa.measure %0 : !qssa.qubit<2> -> tensor<2xi1>
//   return
// }
