#ifndef QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
#define QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "Dialect/Quantum/QuantumOps.td"

def : Pat<(PauliXGateOp (PauliXGateOp $qinp)), (IDGateOp $qinp)>;

#endif // QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
