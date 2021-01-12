
func @CNOT(%0 : node, %1 : node
           %a : node, %b : node) {
  %wl, %wr = wire

  X(%0, %a, %wl)
  Z(%1, %b, %wr)
  /// equiv
  // X(%0, %a, %wr)
  // Z(%1, %b, %wl)

  // or annotate at the end (?)
  // wire %wl %wr

  return
}

func CNOTv2() {
  %p, %q, %r = X
  // wire %0 %p
  // wire %a %q
  %p2, %q2, %r2 = Z
  // wire %1 %p2
  // wire %b %q2
  wire %r  %r2
  return %p %p2 %q %q2
}

// decision: wiring at caller or callee (?)

%... qcall (X, %...)
