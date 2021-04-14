#ifndef CONVERSION_QASMTOSCF_PASSES_H
#define CONVERSION_QASMTOSCF_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createQASMToSCFPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/QASMToSCF/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QASMTOSCF_PASSES_H
