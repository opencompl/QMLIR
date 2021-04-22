#ifndef CONVERSION_QUANTUMTOQASM_PASSES_TD
#define CONVERSION_QUANTUMTOQASM_PASSES_TD

include "mlir/Pass/PassBase.td"

def QuantumToQASMPass : Pass<"convert-qssa-to-qasm", "ModuleOp"> {
  let summary = "Convert qssa dialect to qasm dialect (reg2mem)";
  let constructor = "mlir::createQuantumToQASMPass()";
  let dependentDialects = ["QASM::QASMDialect",
                           "quantum::QuantumDialect",
                           "StandardOpsDialect",
                           "AffineDialect",
                           "scf::SCFDialect"];
}

#endif // CONVERSION_QUANTUMTOQASM_PASSES_TD
