#ifndef QASM_ANALYSIS_PASSDETAIL_H_
#define QASM_ANALYSIS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/QASM/Analysis/Passes.h.inc"

} // end namespace mlir

#endif // QASM_ANALYSIS_PASSDETAIL_H_
