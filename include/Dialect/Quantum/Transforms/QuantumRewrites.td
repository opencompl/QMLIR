#ifndef QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
#define QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "Dialect/Quantum/QuantumOps.td"

// def : Pat<(CNOTGateOp
//             (CNOTGateOp $q0, $q1):$res0__0)
//             (CNOTGateOp $q0, $q1):$res1__1)),
//           [(replaceWithValue $q0), (replaceWithValue $q1)]>;

// S.S = Z
def : Pat<(PhaseGateOp (PhaseGateOp $q)), (PauliZGateOp $q)>;
// S^-1 . S^-1 = Z
def : Pat<(PhaseDaggerGateOp (PhaseDaggerGateOp $q)), (PauliZGateOp $q)>;
// T.T = S
def : Pat<(TGateOp (TGateOp $q)), (PhaseGateOp $q)>;
// T^-1 . T^-1 = S^-1
def : Pat<(TDaggerGateOp (TDaggerGateOp $q)), (PhaseDaggerGateOp $q)>;

#endif // QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
