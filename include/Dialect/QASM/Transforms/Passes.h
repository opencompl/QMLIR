#ifndef QASM_TRANSFORMS_PASSES_H_
#define QASM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<FunctionPass> createQASMMakeGatesOpaquePass();
std::unique_ptr<FunctionPass> createQASMMakeGatesTransparentPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/QASM/Transforms/Passes.h.inc"

} // namespace mlir

#endif // QASM_TRANSFORMS_PASSES_H_
