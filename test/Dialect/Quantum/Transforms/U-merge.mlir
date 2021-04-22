// RUN: quantum-opt --qssa-apply-rewrites %s

func @main() {
  %cst = constant 0.0 : f64
  %0 = qssa.alloc : !qssa.qubit<1>
  %1 = qssa.U(%cst : f64, %cst : f64, %cst : f64) %0 {idx = 1} : !qssa.qubit<1>
  %2 = qssa.U(%cst : f64, %cst : f64, %cst : f64) %1 {idx = 2} : !qssa.qubit<1> 
  %3 = qssa.U(%cst : f64, %cst : f64, %cst : f64) %2 {idx = 3} : !qssa.qubit<1>
  qssa.sink %3 : !qssa.qubit<1>
  return
}
