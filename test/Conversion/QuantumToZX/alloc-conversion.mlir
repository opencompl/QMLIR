// RUN: quantum-opt %s --qssa-prepare-for-zx --convert-qssa-to-zx

func @alloc() {
  %0 = qssa.alloc : !qssa.qubit<1>
  qssa.sink %0 : !qssa.qubit<1>
  return
}

// func @allocMultiqubit() {
//   %0 = qssa.alloc : !qssa.qubit<2>
//   qssa.sink %0 : !qssa.qubit<2>
//   return
// }
