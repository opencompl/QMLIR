#ifndef QUANTUM_TRANSFORMS_PASSES_TD
#define QUANTUM_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def QuantumRewritePass : FunctionPass<"qssa-apply-rewrites"> {
  let summary = "Apply basic qssa rewrite rules";
  let constructor = "mlir::createQuantumRewritePass()";
}

def QuantumDepthComputePass : FunctionPass<"qssa-compute-depths"> {
  let summary = "Compute circuit depth for all operations";
  let constructor = "mlir::createQuantumDepthComputePass()";
}

def QuantumDepthClearPass : FunctionPass<"qssa-clear-depths"> {
  let summary = "Clear all computed depth attributes";
  let constructor = "mlir::createQuantumClearDepthPass()";
}

#endif // QUANTUM_TRANSFORMS_PASSES_TD
