// no-run: quantum-opt %s
// https://github.com/Qiskit/openqasm/blob/fa516eb3c3da4a8546b226e312f0e8bcc791a233/examples/teleport.qasm

module {
  // OPENQASM 3;
  // include "stdgates.inc";
  func @main() -> !qasm.qubit<1> {
    %zero = constant 0 : i64
    %one = constant 1 : i64
    %two = constant 2 : i64
    %three = constant 3 : i64

    %q = std.alloc() : !qasm.qubit<3>
    %c0 = std.alloc() : memref<1xi1>  // bit c0
    %c1 = std.alloc() : memref<1xi1>  // bit c1
    %c2 = std.alloc() : memref<1xi1>  // bit c2

    %q0 = std.subview %q[%zero][%one] : memref<3x!qasm.qubit> -> memref<1x!qasm.qubit>
    %q1 = std.subview %q[%one][%one] : memref<3x!qasm.qubit> -> memref<1x!qasm.qubit>
    %q2 = std.subview %q[%two][%one] : memref<3x!qasm.qubit> -> memref<1x!qasm.qubit>

    qasm.reset(q) // reset q;

    %dot3 = std.constant 0.3 : f64
    %dot2 = std.constant 0.2 : f64
    %dot1 = std.constant 0.1 : f64
    // -- | should this be a function? qasm.u3(%dot3, %dot2, %dot1, %q0) ?
    %u3 = qasm.u3(%dot3, %dot2, %dot1)
    qasm.apply(%u3, %q0); // u3(0.3, 0.2, 0.1) q[0];


    // -- | should these be functions?
    %h = qasm.h  // access gate h
    %cx = qasm.cx // access gate cx
    %z = qasm.z // access gate z
    %x = qasm.x // access gate x

    // | should this be std.call(%h, %q1) ?
    qasm.apply(%h, %q1) // h q[1];
    qasm.apply(%cx, %q1, %q2) // cx q[1], q[2];
    qasm.barrier(%q) // barrier q;
    qasm.apply(%cx, %q0, %q1) cx q[0], q[1];
    qasm.apply(%h, %q0) // h q[0];

    qasm.measure(%c0, %q0) // c0 = measure q[0];
    qasm.measure(%c1, %q1) // c1 = measure q[1];

    %c0_eq_1 = std.cmpi "eq" %c0, %one
    scf.if %c0_eq_1 {
        qasm.apply(%z, q2)
    } // if(c0==1) z q[2];

    %c1_eq_1 = std.cmpi "eq" %c1, %one
    scf.if %c1_eq_1 {
        qasm.apply(%x, q2)
    } // if(c1==1) { x q[2]; }
    qasm.measure(%c2, q[2]);
    std.return(%c2)
  }
}

