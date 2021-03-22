#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/Quantum/QuantumOps.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::quantum;

class QuantumRewritePass : public QuantumRewritePassBase<QuantumRewritePass> {
  void runOnFunction() override;
};

namespace {
#include "Dialect/Quantum/Transforms/QuantumRewrites.h.inc"
}

void QuantumRewritePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  populateWithGenerated(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumRewritePass() {
  return std::make_unique<QuantumRewritePass>();
}

} // namespace mlir
