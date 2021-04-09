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

class GateCountPass : public QuantumGateCountPassBase<GateCountPass> {
  void runOnFunction() override;
};

void GateCountPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  // patterns.insert<
  //     // clang-format off
  //     // clang-format on
  //     >(&getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumGateCountPass() {
  return std::make_unique<GateCountPass>();
}

} // namespace mlir
