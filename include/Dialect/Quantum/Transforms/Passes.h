#ifndef QUANTUM_TRANSFORMS_PASSES_H_
#define QUANTUM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<FunctionPass> createQuantumRewritePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir

#endif // QUANTUM_TRANSFORMS_PASSES_H_
