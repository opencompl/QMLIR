OPENQASM 2.0;

qreg q[4];
creg c[4];
measure q -> c;
if (c == 0) U(0,0,0) q;
