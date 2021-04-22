#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"

using namespace mlir;
using namespace mlir::quantum;

class PrepareForZXPass : public QuantumPrepareForZXPassBase<PrepareForZXPass> {
  void runOnFunction() override;
};

namespace {
#include "Dialect/Quantum/Transforms/PrepareForZX.h.inc"
} // namespace

void PrepareForZXPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  populateWithGenerated(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumPrepareForZXPass() {
  return std::make_unique<PrepareForZXPass>();
}

} // namespace mlir
