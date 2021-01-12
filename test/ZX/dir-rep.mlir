
func @CNOT(%0: node, %1 : node) {
  // inputs %0 %1
  %a, %m = Z(%0)
  %b = X(%m, %1)
  // outputs %a %b

  return %a, %b
}

func @CNOT(%0: node, %1 : node) {
  // inputs %0 %1
  %b, %n = X(%1)
  %a = Z(%n, %0)
  // outputs %a %b

  return %a, %b
}
