#ifndef QASM_ANALYSIS_PASSES
#define QASM_ANALYSIS_PASSES

include "mlir/Pass/PassBase.td"

def GateCountPass : Pass<"qasm-gate-count", "ModuleOp"> {
  let summary = "QASM Gate Count analysis pass";
  let constructor = "mlir::createQASMGateCountPass()";
}

#endif // QASM_ANALYSIS_PASSES
