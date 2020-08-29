#ifndef CONVERSION_QUANTUMTOSTANDARD_PASSDETAIL_H_
#define CONVERSION_QUANTUMTOSTANDARD_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "QuantumToStandard/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_QUANTUMTOSTANDARD_PASSDETAIL_H_
