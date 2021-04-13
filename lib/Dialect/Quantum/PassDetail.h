#ifndef QUANTUM_PASSDETAIL_H_
#define QUANTUM_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/Quantum/Passes.h.inc"

} // end namespace mlir

#endif // QUANTUM_PASSDETAIL_H_
