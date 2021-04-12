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

class GateCountAnalysis {
  llvm::StringMap<int> gateCounts;

  void addGate(FuncOp func, StringRef gateName) { gateCounts[gateName] += 1; }

public:
  GateCountAnalysis(Operation *op) {
    auto module = dyn_cast<ModuleOp>(op);

    module.walk([&](Operation *op) {
      if (auto cnotOp = dyn_cast<CNOTGateOp>(op)) {
        auto func = cnotOp->getParentOfType<FuncOp>();
        addGate(func, "CNOT");
      }
    });
  }
};

void GateCountPass::runOnFunction() {
  markAllAnalysesPreserved();
  auto analysis = getAnalysis<GateCountAnalysis>();
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumGateCountPass() {
  return std::make_unique<GateCountPass>();
}

} // namespace mlir
