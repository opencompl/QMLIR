#ifndef CONVERSION_QUANTUMTOQASM_PASSES_H
#define CONVERSION_QUANTUMTOQASM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createQuantumToQASMPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/QuantumToQASM/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_QUANTUMTOQASM_PASSES_H
