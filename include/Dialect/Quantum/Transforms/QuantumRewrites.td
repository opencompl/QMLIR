#ifndef QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
#define QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "Dialect/Quantum/QuantumOps.td"

// def : Pat<(CNOTGateOp
//             (CNOTGateOp $q0, $q1):$res0__0)
//             (CNOTGateOp $q0, $q1):$res1__1)),
//           [(replaceWithValue $q0), (replaceWithValue $q1)]>;

#endif // QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
