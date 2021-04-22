#ifndef CONVERSION_QUANTUMTOZX_PASSES_H
#define CONVERSION_QUANTUMTOZX_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createQuantumToZXPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/QuantumToZX/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QUANTUMTOZX_PASSES_H
