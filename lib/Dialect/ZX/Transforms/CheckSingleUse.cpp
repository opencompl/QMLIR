// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
//
#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::ZX;

class ZXCheckSingleUsePass
    : public ZXCheckSingleUsePassBase<ZXCheckSingleUsePass> {
  void runOnFunction() override;
};

void ZXCheckSingleUsePass::runOnFunction() {
  FuncOp f = getOperation();
  f.walk([](Operation *op) {
    for (auto res : op->getResults()) {
      if (res.getType().isa<WireType>() && !res.hasOneUse()) {
        if (res.getUses().empty()) {
          emitError(res.getLoc()) << "ZX Wire declared here is not used. "
                                     "Perhaps missing `zx.sink`?";
        } else {
          emitError(res.getLoc())
              << "ZX Wire declared here is used multiple times.";
          for (auto *user : res.getUsers()) {
            emitError(user->getLoc()) << "used here";
          }
        }
      }
    }
  });
}

namespace mlir {

std::unique_ptr<FunctionPass> createZXCheckSingleUsePass() {
  return std::make_unique<ZXCheckSingleUsePass>();
}

} // namespace mlir
