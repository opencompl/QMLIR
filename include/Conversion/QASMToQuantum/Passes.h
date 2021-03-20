#ifndef CONVERSION_QASMTOQUANTUM_PASSES_H
#define CONVERSION_QASMTOQUANTUM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createQASMToQuantumPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/QASMToQuantum/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QASMTOQUANTUM_PASSES_H
