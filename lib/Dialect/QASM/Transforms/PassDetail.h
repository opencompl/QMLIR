#ifndef QASM_TRANSFORMS_PASSDETAIL_H_
#define QASM_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/QASM/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // QASM_TRANSFORMS_PASSDETAIL_H_
