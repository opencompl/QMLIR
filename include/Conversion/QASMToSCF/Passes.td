#ifndef CONVERSION_QASMTOSCF_PASSES_TD
#define CONVERSION_QASMTOSCF_PASSES_TD

include "mlir/Pass/PassBase.td"

def QASMToSCFPass : Pass<"convert-qasm-to-scf", "ModuleOp"> {
  let summary = "Convert QASM dialect control flow to scf dialect";
  let constructor = "mlir::createQASMToSCFPass()";
  let dependentDialects = ["QASM::QASMDialect", "scf::SCFDialect",
                           "AffineDialect", "StandardOpsDialect"];
}

#endif // CONVERSION_QASMTOSCF_PASSES_TD
