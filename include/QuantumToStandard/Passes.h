#ifndef CONVERSION_QUANTUMTOSTANDARD_PASSES_H
#define CONVERSION_QUANTUMTOSTANDARD_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertQuantumToStandardPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "QuantumToStandard/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QUANTUMTOSTANDARD_PASSES_H
