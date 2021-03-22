// RUN: quantum-opt %s | quantum-opt

module {
  // OPENQASM 2.0;
  // include "qelib1.inc";
  // https://github.com/Qiskit/openqasm/blob/OpenQASM2.x/examples/qelib1.inc

  // gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
  func @u3(%theta : f32, %phi : f32, %lambda : f32, %q: !qasm.qubit) {
    qasm.U(%theta : f32, %phi : f32, %lambda : f32) %q
    return
  }
  // gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
  func @u2(%phi : f32, %lambda : f32, %q: !qasm.qubit) {
    %pi = qasm.pi : f32
    %2 = constant 2.0 : f32
    %piby2 = divf %pi, %2 : f32
    qasm.U(%piby2 : f32, %phi : f32, %lambda : f32) %q
    return
  }
  // gate u1(lambda) q { U(0,0,lambda) q; }
  func @u1(%lambda : f32, %q: !qasm.qubit) {
    %0 = constant 0.0 : f32
    qasm.U(%0 : f32, %0 : f32, %lambda : f32) %q
    return
  }

  // gate x a { u3(pi,0,pi) a; }
  func @x(%a : !qasm.qubit) {
    %0 = constant 0.0 : f32
    %pi = qasm.pi : f32
    call @u3(%pi, %0, %pi, %a) : (f32, f32, f32, !qasm.qubit) -> ()
    return
  }
  // gate z a { u1(pi) a; }
  func @z(%a : !qasm.qubit) {
    %pi = qasm.pi : f32
    call @u1(%pi, %a) : (f32, !qasm.qubit) -> ()
    return
  }

  // gate h a { u2(0,pi) a; }
  func @h(%a : !qasm.qubit) {
    %0 = constant 0.0 : f32
    %pi = qasm.pi : f32
    call @u2(%0, %pi, %a) : (f32, f32, !qasm.qubit) -> ()
    return
  }

  // https://github.com/Qiskit/openqasm/blob/OpenQASM2.x/examples/teleport.qasm
  // optional post-rotation for state tomography
  // gate post q { }
  func @post(%q : !qasm.qubit) {
    return
  }
  func @main() {
    %0 = constant 0 : index
    %1 = constant 1 : index
    %2 = constant 2 : index
    %3 = constant 3 : index

    // qreg q[3];
    %q0 = qasm.allocate
    %q1 = qasm.allocate
    %q2 = qasm.allocate

    // creg c0[1];
    %c0 = memref.alloc() : memref<1xi1>
    // creg c1[1];
    %c1 = memref.alloc() : memref<1xi1>
    // creg c2[1];
    %c2 = memref.alloc() : memref<1xi1>

    // u3(0.3,0.2,0.1) q[0];
    %cst0 = constant 0.3 : f32
    %cst1 = constant 0.2 : f32
    %cst2 = constant 0.1 : f32
    call @u3(%cst0, %cst1, %cst2, %q0) : (f32, f32, f32, !qasm.qubit) -> ()

    // h q[1];
    call @h(%q1) : (!qasm.qubit) -> ()

    // cx q[1],q[2];
    qasm.CX %q1, %q2

    // barrier q;
    qasm.barrier %q0
    qasm.barrier %q1
    qasm.barrier %q2

    // cx q[0],q[1];
    qasm.CX %q0, %q1

    // h q[0];
    call @h(%q0) : (!qasm.qubit) -> ()

    // measure q[0] -> c0[0];
    %tmp0 = qasm.measure %q0
    memref.store %tmp0, %c0[%0] : memref<1xi1>

    // measure q[1] -> c1[0];
    %tmp1 = qasm.measure %q1
    memref.store %tmp1, %c1[%0] : memref<1xi1>

    // if(c0==1) z q[2];
    %cond0 = memref.load %c0[%0] : memref<1xi1>
    scf.if %cond0 {
      call @z(%q2) : (!qasm.qubit) -> ()
    }

    // if(c1==1) x q[2];
    %cond1 = memref.load %c1[%0] : memref<1xi1>
    scf.if %cond1 {
      call @x(%q2) : (!qasm.qubit) -> ()
    }

    // post q[2];
    call @post(%q2) : (!qasm.qubit) -> ()

    // measure q[2] -> c2[0];
    %tmp2 = qasm.measure %q2
    memref.store %tmp2, %c2[%0] : memref<1xi1>

    return
  }
}

