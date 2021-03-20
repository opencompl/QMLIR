#ifndef QUANTUM_TRANSFORMS_PASSES_TD
#define QUANTUM_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def QuantumRewritePass : FunctionPass<"apply-qssa-rewrites"> {
  let summary = "Apply basic qssa rewrite rules";
  let constructor = "mlir::createQuantumRewritePass()";
}

#endif // QUANTUM_TRANSFORMS_PASSES_TD
