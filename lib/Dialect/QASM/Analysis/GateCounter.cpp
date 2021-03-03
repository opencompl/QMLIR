#include "Dialect/QASM/Analysis/GateCounter.h"
#include "Dialect/QASM/Analysis/Passes.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

class GateCountPass : public GateCountPassBase<GateCountPass> {
  void runOnOperation() override;
};

void GateCountPass::runOnOperation() { ModuleOp module = getOperation(); }

} // namespace

namespace mlir {
std::unique_ptr<Pass> createQASMGateCountPass() {
  return std::make_unique<GateCountPass>();
}
} // namespace mlir