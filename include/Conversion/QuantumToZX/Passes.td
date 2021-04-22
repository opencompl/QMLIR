#ifndef CONVERSION_QUANTUMTOZX_PASSES_TD
#define CONVERSION_QUANTUMTOZX_PASSES_TD

include "mlir/Pass/PassBase.td"

def QuantumToZXPass : Pass<"convert-qssa-to-zx", "ModuleOp"> {
  let summary = "Convert qssa dialect to zx dialect";
  let constructor = "mlir::createQuantumToZXPass()";
  let dependentDialects = ["ZX::ZXDialect",
                           "quantum::QuantumDialect",
                           "StandardOpsDialect"];
}

#endif // CONVERSION_QUANTUMTOZX_PASSES_TD
