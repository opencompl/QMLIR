#ifndef CONVERSION_QASMTOQUANTUM_PASSES_TD
#define CONVERSION_QASMTOQUANTUM_PASSES_TD

include "mlir/Pass/PassBase.td"

def QASMToQuantumPass : Pass<"convert-qasm-to-qssa", "ModuleOp"> {
  let summary = "Convert QASM dialect to quantum/qssa dialect";
  let constructor = "mlir::createQASMToQuantumPass()";
  let dependentDialects = ["QASM::QASMDialect",
                           "quantum::QuantumDialect",
                           "StandardOpsDialect",
                           "AffineDialect",
                           "scf::SCFDialect"];
}

#endif // CONVERSION_QASMTOQUANTUM_PASSES_TD
