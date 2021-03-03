#ifndef QASM_ANALYSIS_PASSES_H
#define QASM_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createQASMGateCountPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/QASM/Analysis/Passes.h.inc"

} // namespace mlir

#endif