#ifndef CONVERSION_ZXTOQUANTUM_PASSES_H
#define CONVERSION_ZXTOQUANTUM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createZXToQuantumPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/ZXToQuantum/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_ZXTOQUANTUM_PASSES_H
