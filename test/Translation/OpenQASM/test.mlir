module {
  func @u3 (%0 : f32, %1 : f32, %2 : f32, %3 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.u3} {
    qasm.U (%0 : f32, %1 : f32, %2 : f32) %3
    std.return {qasm.gate_end}
  }
  func @u2 (%0 : f32, %1 : f32, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.u2} {
    %3 = qasm.pi : f32
    %4 = std.constant 2 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = std.divf %3, %5 : f32
    qasm.U (%6 : f32, %0 : f32, %1 : f32) %2
    std.return {qasm.gate_end}
  }
  func @u1 (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.u1} {
    %2 = std.constant 0 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.constant 0 : i32
    %5 = std.sitofp %4 : i32 to f32
    qasm.U (%3 : f32, %5 : f32, %0 : f32) %1
    std.return {qasm.gate_end}
  }
  func @cx (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cx} {
    qasm.CX %0, %1
    std.return {qasm.gate_end}
  }
  func @id (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.id} {
    %1 = std.constant 0 : i32
    %2 = std.sitofp %1 : i32 to f32
    %3 = std.constant 0 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.constant 0 : i32
    %6 = std.sitofp %5 : i32 to f32
    qasm.U (%2 : f32, %4 : f32, %6 : f32) %0
    std.return {qasm.gate_end}
  }
  func @u0 (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.u0} {
    %2 = std.constant 0 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.constant 0 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = std.constant 0 : i32
    %7 = std.sitofp %6 : i32 to f32
    qasm.U (%3 : f32, %5 : f32, %7 : f32) %1
    std.return {qasm.gate_end}
  }
  func @u (%0 : f32, %1 : f32, %2 : f32, %3 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.u} {
    qasm.U (%0 : f32, %1 : f32, %2 : f32) %3
    std.return {qasm.gate_end}
  }
  func @p (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.p} {
    %2 = std.constant 0 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.constant 0 : i32
    %5 = std.sitofp %4 : i32 to f32
    qasm.U (%3 : f32, %5 : f32, %0 : f32) %1
    std.return {qasm.gate_end}
  }
  func @x (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.x} {
    %1 = qasm.pi : f32
    %2 = std.constant 0 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = qasm.pi : f32
    std.call @u3(%1, %3, %4, %0) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @y (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.y} {
    %1 = qasm.pi : f32
    %2 = qasm.pi : f32
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %2, %4 : f32
    %6 = qasm.pi : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @u3(%1, %5, %9, %0) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @z (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.z} {
    %1 = qasm.pi : f32
    std.call @u1(%1, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @h (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.h} {
    %1 = std.constant 0 : i32
    %2 = std.sitofp %1 : i32 to f32
    %3 = qasm.pi : f32
    std.call @u2(%2, %3, %0) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @s (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.s} {
    %1 = qasm.pi : f32
    %2 = std.constant 2 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.divf %1, %3 : f32
    std.call @u1(%4, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @sdg (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.sdg} {
    %1 = qasm.pi : f32
    %2 = std.negf %1 : f32
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %2, %4 : f32
    std.call @u1(%5, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @t (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.t} {
    %1 = qasm.pi : f32
    %2 = std.constant 4 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.divf %1, %3 : f32
    std.call @u1(%4, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @tdg (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.tdg} {
    %1 = qasm.pi : f32
    %2 = std.negf %1 : f32
    %3 = std.constant 4 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %2, %4 : f32
    std.call @u1(%5, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rx (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rx} {
    %2 = qasm.pi : f32
    %3 = std.negf %2 : f32
    %4 = std.constant 2 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = std.divf %3, %5 : f32
    %7 = qasm.pi : f32
    %8 = std.constant 2 : i32
    %9 = std.sitofp %8 : i32 to f32
    %10 = std.divf %7, %9 : f32
    std.call @u3(%0, %6, %10, %1) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @ry (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.ry} {
    %2 = std.constant 0 : i32
    %3 = std.sitofp %2 : i32 to f32
    %4 = std.constant 0 : i32
    %5 = std.sitofp %4 : i32 to f32
    std.call @u3(%0, %3, %5, %1) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rz (%0 : f32, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rz} {
    std.call @u1(%0, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @sx (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.sx} {
    std.call @sdg(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @h(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @sdg(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @sxdg (%0 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.sxdg} {
    std.call @s(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @h(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @s(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cz (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cz} {
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cy (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cy} {
    std.call @sdg(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @s(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @swap (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.swap} {
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%1, %0) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @ch (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.ch} {
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @sdg(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @t(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @t(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @s(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @x(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @s(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @ccx (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.ccx} {
    std.call @h(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @tdg(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @tdg(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @t(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @t(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @h(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @t(%0) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @tdg(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cswap (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cswap} {
    std.call @cx(%2, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @ccx(%0, %1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%2, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @crx (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.crx} {
    %3 = qasm.pi : f32
    %4 = std.constant 2 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = std.divf %3, %5 : f32
    std.call @u1(%6, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %7 = std.negf %0 : f32
    %8 = std.constant 2 : i32
    %9 = std.sitofp %8 : i32 to f32
    %10 = std.divf %7, %9 : f32
    %11 = std.constant 0 : i32
    %12 = std.sitofp %11 : i32 to f32
    %13 = std.constant 0 : i32
    %14 = std.sitofp %13 : i32 to f32
    std.call @u3(%10, %12, %14, %2) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %15 = std.constant 2 : i32
    %16 = std.sitofp %15 : i32 to f32
    %17 = std.divf %0, %16 : f32
    %18 = qasm.pi : f32
    %19 = std.negf %18 : f32
    %20 = std.constant 2 : i32
    %21 = std.sitofp %20 : i32 to f32
    %22 = std.divf %19, %21 : f32
    %23 = std.constant 0 : i32
    %24 = std.sitofp %23 : i32 to f32
    std.call @u3(%17, %22, %24, %2) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cry (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cry} {
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %0, %4 : f32
    %6 = std.constant 0 : i32
    %7 = std.sitofp %6 : i32 to f32
    %8 = std.constant 0 : i32
    %9 = std.sitofp %8 : i32 to f32
    std.call @u3(%5, %7, %9, %2) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %10 = std.negf %0 : f32
    %11 = std.constant 2 : i32
    %12 = std.sitofp %11 : i32 to f32
    %13 = std.divf %10, %12 : f32
    %14 = std.constant 0 : i32
    %15 = std.sitofp %14 : i32 to f32
    %16 = std.constant 0 : i32
    %17 = std.sitofp %16 : i32 to f32
    std.call @u3(%13, %15, %17, %2) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @crz (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.crz} {
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %0, %4 : f32
    std.call @u1(%5, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %6 = std.negf %0 : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @u1(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cu1 (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cu1} {
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %0, %4 : f32
    std.call @u1(%5, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %6 = std.negf %0 : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @u1(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %10 = std.constant 2 : i32
    %11 = std.sitofp %10 : i32 to f32
    %12 = std.divf %0, %11 : f32
    std.call @u1(%12, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cp (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cp} {
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %0, %4 : f32
    std.call @p(%5, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %6 = std.negf %0 : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @p(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %10 = std.constant 2 : i32
    %11 = std.sitofp %10 : i32 to f32
    %12 = std.divf %0, %11 : f32
    std.call @p(%12, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cu3 (%0 : f32, %1 : f32, %2 : f32, %3 : !qasm.qubit, %4 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cu3} {
    %5 = std.addf %2, %1 : f32
    %6 = std.constant 2 : i32
    %7 = std.sitofp %6 : i32 to f32
    %8 = std.divf %5, %7 : f32
    std.call @u1(%8, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %9 = std.subf %2, %1 : f32
    %10 = std.constant 2 : i32
    %11 = std.sitofp %10 : i32 to f32
    %12 = std.divf %9, %11 : f32
    std.call @u1(%12, %4) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%3, %4) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %13 = std.negf %0 : f32
    %14 = std.constant 2 : i32
    %15 = std.sitofp %14 : i32 to f32
    %16 = std.divf %13, %15 : f32
    %17 = std.constant 0 : i32
    %18 = std.sitofp %17 : i32 to f32
    %19 = std.addf %1, %2 : f32
    %20 = std.negf %19 : f32
    %21 = std.constant 2 : i32
    %22 = std.sitofp %21 : i32 to f32
    %23 = std.divf %20, %22 : f32
    std.call @u3(%16, %18, %23, %4) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @cx(%3, %4) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %24 = std.constant 2 : i32
    %25 = std.sitofp %24 : i32 to f32
    %26 = std.divf %0, %25 : f32
    %27 = std.constant 0 : i32
    %28 = std.sitofp %27 : i32 to f32
    std.call @u3(%26, %1, %28, %4) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @csx (%0 : !qasm.qubit, %1 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.csx} {
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    %2 = qasm.pi : f32
    %3 = std.constant 2 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = std.divf %2, %4 : f32
    std.call @cu1(%5, %0, %1) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%1) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @cu (%0 : f32, %1 : f32, %2 : f32, %3 : f32, %4 : !qasm.qubit, %5 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.cu} {
    std.call @p(%3, %4) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %6 = std.addf %2, %1 : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @p(%9, %4) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %10 = std.subf %2, %1 : f32
    %11 = std.constant 2 : i32
    %12 = std.sitofp %11 : i32 to f32
    %13 = std.divf %10, %12 : f32
    std.call @p(%13, %5) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%4, %5) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %14 = std.negf %0 : f32
    %15 = std.constant 2 : i32
    %16 = std.sitofp %15 : i32 to f32
    %17 = std.divf %14, %16 : f32
    %18 = std.constant 0 : i32
    %19 = std.sitofp %18 : i32 to f32
    %20 = std.addf %1, %2 : f32
    %21 = std.negf %20 : f32
    %22 = std.constant 2 : i32
    %23 = std.sitofp %22 : i32 to f32
    %24 = std.divf %21, %23 : f32
    std.call @u(%17, %19, %24, %5) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @cx(%4, %5) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %25 = std.constant 2 : i32
    %26 = std.sitofp %25 : i32 to f32
    %27 = std.divf %0, %26 : f32
    %28 = std.constant 0 : i32
    %29 = std.sitofp %28 : i32 to f32
    std.call @u(%27, %1, %29, %5) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rxx (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rxx} {
    %3 = qasm.pi : f32
    %4 = std.constant 2 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = std.divf %3, %5 : f32
    %7 = std.constant 0 : i32
    %8 = std.sitofp %7 : i32 to f32
    std.call @u3(%6, %0, %8, %1) {qasm.gate} : (f32, f32, f32, !qasm.qubit) -> ()
    std.call @h(%2) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %9 = std.negf %0 : f32
    std.call @u1(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%2) {qasm.gate} : (!qasm.qubit) -> ()
    %10 = qasm.pi : f32
    %11 = std.negf %10 : f32
    %12 = qasm.pi : f32
    %13 = std.subf %12, %0 : f32
    std.call @u2(%11, %13, %1) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rzz (%0 : f32, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rzz} {
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @u1(%0, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rccx (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rccx} {
    %3 = std.constant 0 : i32
    %4 = std.sitofp %3 : i32 to f32
    %5 = qasm.pi : f32
    std.call @u2(%4, %5, %2) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    %6 = qasm.pi : f32
    %7 = std.constant 4 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @u1(%9, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %10 = qasm.pi : f32
    %11 = std.negf %10 : f32
    %12 = std.constant 4 : i32
    %13 = std.sitofp %12 : i32 to f32
    %14 = std.divf %11, %13 : f32
    std.call @u1(%14, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %15 = qasm.pi : f32
    %16 = std.constant 4 : i32
    %17 = std.sitofp %16 : i32 to f32
    %18 = std.divf %15, %17 : f32
    std.call @u1(%18, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %19 = qasm.pi : f32
    %20 = std.negf %19 : f32
    %21 = std.constant 4 : i32
    %22 = std.sitofp %21 : i32 to f32
    %23 = std.divf %20, %22 : f32
    std.call @u1(%23, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %24 = std.constant 0 : i32
    %25 = std.sitofp %24 : i32 to f32
    %26 = qasm.pi : f32
    std.call @u2(%25, %26, %2) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @rc3x (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit, %3 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.rc3x} {
    %4 = std.constant 0 : i32
    %5 = std.sitofp %4 : i32 to f32
    %6 = qasm.pi : f32
    std.call @u2(%5, %6, %3) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    %7 = qasm.pi : f32
    %8 = std.constant 4 : i32
    %9 = std.sitofp %8 : i32 to f32
    %10 = std.divf %7, %9 : f32
    std.call @u1(%10, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %11 = qasm.pi : f32
    %12 = std.negf %11 : f32
    %13 = std.constant 4 : i32
    %14 = std.sitofp %13 : i32 to f32
    %15 = std.divf %12, %14 : f32
    std.call @u1(%15, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %16 = std.constant 0 : i32
    %17 = std.sitofp %16 : i32 to f32
    %18 = qasm.pi : f32
    std.call @u2(%17, %18, %3) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    std.call @cx(%0, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %19 = qasm.pi : f32
    %20 = std.constant 4 : i32
    %21 = std.sitofp %20 : i32 to f32
    %22 = std.divf %19, %21 : f32
    std.call @u1(%22, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %23 = qasm.pi : f32
    %24 = std.negf %23 : f32
    %25 = std.constant 4 : i32
    %26 = std.sitofp %25 : i32 to f32
    %27 = std.divf %24, %26 : f32
    std.call @u1(%27, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %28 = qasm.pi : f32
    %29 = std.constant 4 : i32
    %30 = std.sitofp %29 : i32 to f32
    %31 = std.divf %28, %30 : f32
    std.call @u1(%31, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %32 = qasm.pi : f32
    %33 = std.negf %32 : f32
    %34 = std.constant 4 : i32
    %35 = std.sitofp %34 : i32 to f32
    %36 = std.divf %33, %35 : f32
    std.call @u1(%36, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %37 = std.constant 0 : i32
    %38 = std.sitofp %37 : i32 to f32
    %39 = qasm.pi : f32
    std.call @u2(%38, %39, %3) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    %40 = qasm.pi : f32
    %41 = std.constant 4 : i32
    %42 = std.sitofp %41 : i32 to f32
    %43 = std.divf %40, %42 : f32
    std.call @u1(%43, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %44 = qasm.pi : f32
    %45 = std.negf %44 : f32
    %46 = std.constant 4 : i32
    %47 = std.sitofp %46 : i32 to f32
    %48 = std.divf %45, %47 : f32
    std.call @u1(%48, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %49 = std.constant 0 : i32
    %50 = std.sitofp %49 : i32 to f32
    %51 = qasm.pi : f32
    std.call @u2(%50, %51, %3) {qasm.gate} : (f32, f32, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @c3x (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit, %3 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.c3x} {
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %4 = qasm.pi : f32
    %5 = std.constant 8 : i32
    %6 = std.sitofp %5 : i32 to f32
    %7 = std.divf %4, %6 : f32
    std.call @p(%7, %0) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %8 = qasm.pi : f32
    %9 = std.constant 8 : i32
    %10 = std.sitofp %9 : i32 to f32
    %11 = std.divf %8, %10 : f32
    std.call @p(%11, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %12 = qasm.pi : f32
    %13 = std.constant 8 : i32
    %14 = std.sitofp %13 : i32 to f32
    %15 = std.divf %12, %14 : f32
    std.call @p(%15, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    %16 = qasm.pi : f32
    %17 = std.constant 8 : i32
    %18 = std.sitofp %17 : i32 to f32
    %19 = std.divf %16, %18 : f32
    std.call @p(%19, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %20 = qasm.pi : f32
    %21 = std.negf %20 : f32
    %22 = std.constant 8 : i32
    %23 = std.sitofp %22 : i32 to f32
    %24 = std.divf %21, %23 : f32
    std.call @p(%24, %1) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %25 = qasm.pi : f32
    %26 = std.negf %25 : f32
    %27 = std.constant 8 : i32
    %28 = std.sitofp %27 : i32 to f32
    %29 = std.divf %26, %28 : f32
    std.call @p(%29, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %30 = qasm.pi : f32
    %31 = std.constant 8 : i32
    %32 = std.sitofp %31 : i32 to f32
    %33 = std.divf %30, %32 : f32
    std.call @p(%33, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %34 = qasm.pi : f32
    %35 = std.negf %34 : f32
    %36 = std.constant 8 : i32
    %37 = std.sitofp %36 : i32 to f32
    %38 = std.divf %35, %37 : f32
    std.call @p(%38, %2) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %39 = qasm.pi : f32
    %40 = std.negf %39 : f32
    %41 = std.constant 8 : i32
    %42 = std.sitofp %41 : i32 to f32
    %43 = std.divf %40, %42 : f32
    std.call @p(%43, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %44 = qasm.pi : f32
    %45 = std.constant 8 : i32
    %46 = std.sitofp %45 : i32 to f32
    %47 = std.divf %44, %46 : f32
    std.call @p(%47, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %48 = qasm.pi : f32
    %49 = std.negf %48 : f32
    %50 = std.constant 8 : i32
    %51 = std.sitofp %50 : i32 to f32
    %52 = std.divf %49, %51 : f32
    std.call @p(%52, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %53 = qasm.pi : f32
    %54 = std.constant 8 : i32
    %55 = std.sitofp %54 : i32 to f32
    %56 = std.divf %53, %55 : f32
    std.call @p(%56, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %57 = qasm.pi : f32
    %58 = std.negf %57 : f32
    %59 = std.constant 8 : i32
    %60 = std.sitofp %59 : i32 to f32
    %61 = std.divf %58, %60 : f32
    std.call @p(%61, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%1, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %62 = qasm.pi : f32
    %63 = std.constant 8 : i32
    %64 = std.sitofp %63 : i32 to f32
    %65 = std.divf %62, %64 : f32
    std.call @p(%65, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    %66 = qasm.pi : f32
    %67 = std.negf %66 : f32
    %68 = std.constant 8 : i32
    %69 = std.sitofp %68 : i32 to f32
    %70 = std.divf %67, %69 : f32
    std.call @p(%70, %3) {qasm.gate} : (f32, !qasm.qubit) -> ()
    std.call @cx(%0, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @c3sqrtx (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit, %3 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.c3sqrtx} {
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %4 = qasm.pi : f32
    %5 = std.negf %4 : f32
    %6 = std.constant 8 : i32
    %7 = std.sitofp %6 : i32 to f32
    %8 = std.divf %5, %7 : f32
    std.call @cu1(%8, %0, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %9 = qasm.pi : f32
    %10 = std.constant 8 : i32
    %11 = std.sitofp %10 : i32 to f32
    %12 = std.divf %9, %11 : f32
    std.call @cu1(%12, %1, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %13 = qasm.pi : f32
    %14 = std.negf %13 : f32
    %15 = std.constant 8 : i32
    %16 = std.sitofp %15 : i32 to f32
    %17 = std.divf %14, %16 : f32
    std.call @cu1(%17, %1, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %18 = qasm.pi : f32
    %19 = std.constant 8 : i32
    %20 = std.sitofp %19 : i32 to f32
    %21 = std.divf %18, %20 : f32
    std.call @cu1(%21, %2, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %22 = qasm.pi : f32
    %23 = std.negf %22 : f32
    %24 = std.constant 8 : i32
    %25 = std.sitofp %24 : i32 to f32
    %26 = std.divf %23, %25 : f32
    std.call @cu1(%26, %2, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %27 = qasm.pi : f32
    %28 = std.constant 8 : i32
    %29 = std.sitofp %28 : i32 to f32
    %30 = std.divf %27, %29 : f32
    std.call @cu1(%30, %2, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @cx(%0, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    %31 = qasm.pi : f32
    %32 = std.negf %31 : f32
    %33 = std.constant 8 : i32
    %34 = std.sitofp %33 : i32 to f32
    %35 = std.divf %32, %34 : f32
    std.call @cu1(%35, %2, %3) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%3) {qasm.gate} : (!qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @c4x (%0 : !qasm.qubit, %1 : !qasm.qubit, %2 : !qasm.qubit, %3 : !qasm.qubit, %4 : !qasm.qubit) -> () attributes {qasm.gate, qasm.stdgate.c4x} {
    std.call @h(%4) {qasm.gate} : (!qasm.qubit) -> ()
    %5 = qasm.pi : f32
    %6 = std.negf %5 : f32
    %7 = std.constant 2 : i32
    %8 = std.sitofp %7 : i32 to f32
    %9 = std.divf %6, %8 : f32
    std.call @cu1(%9, %3, %4) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%4) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @c3x(%0, %1, %2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%4) {qasm.gate} : (!qasm.qubit) -> ()
    %10 = qasm.pi : f32
    %11 = std.constant 2 : i32
    %12 = std.sitofp %11 : i32 to f32
    %13 = std.divf %10, %12 : f32
    std.call @cu1(%13, %3, %4) {qasm.gate} : (f32, !qasm.qubit, !qasm.qubit) -> ()
    std.call @h(%4) {qasm.gate} : (!qasm.qubit) -> ()
    std.call @c3x(%0, %1, %2, %3) {qasm.gate} : (!qasm.qubit, !qasm.qubit, !qasm.qubit, !qasm.qubit) -> ()
    std.call @c3sqrtx(%0, %1, %2, %4) {qasm.gate} : (!qasm.qubit, !qasm.qubit, !qasm.qubit, !qasm.qubit) -> ()
    std.return {qasm.gate_end}
  }
  func @qasm_main () -> ()  {
    %0 = qasm.allocate 
    %1 = qasm.allocate 
    %2 = qasm.allocate 
    %3 = qasm.allocate 
    %4 = qasm.allocate 
    %5 = qasm.allocate 
    %6 = qasm.allocate 
    %7 = qasm.allocate 
    %8 = qasm.allocate 
    %9 = qasm.allocate 
    %10 = qasm.allocate 
    %11 = qasm.allocate 
    %12 = qasm.allocate 
    %13 = qasm.allocate 
    %14 = qasm.allocate 
    %15 = qasm.allocate 
    %16 = memref.alloc () : memref<16xi1>
    std.call @cx(%1, %2) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.call @cx(%3, %1) {qasm.gate} : (!qasm.qubit, !qasm.qubit) -> ()
    std.return 
  }
}
