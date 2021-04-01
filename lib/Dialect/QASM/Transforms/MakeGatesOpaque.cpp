#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/QASM/QASMOps.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::QASM;

namespace {

class GateCallRewrite : public OpRewritePattern<CallOp> {
  SmallVector<StringRef> gates;

public:
  GateCallRewrite(SmallVector<StringRef> gates, MLIRContext *ctx,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), gates(gates) {}
  LogicalResult match(CallOp op) const override {
    if (!op->hasAttrOfType<UnitAttr>("qasm.gate"))
      return failure();
    for (auto gate : gates) {
      if (gate == op.getCallee())
        return success();
    }
    return failure();
  }
  void rewrite(CallOp op, PatternRewriter &rewriter) const override {
    rewriter.create<GateCall>(op->getLoc(), op.calleeAttr(),
                              op.getArgOperands());
    rewriter.eraseOp(op);
  }
};

class GatePass : public QASMMakeGatesOpaquePassBase<GatePass> {
  void runOnFunction() override;
};

void GatePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  SmallVector<StringRef> gateList(gates.begin(), gates.end());
  patterns.insert<GateCallRewrite>(gateList, &getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
std::unique_ptr<FunctionPass> createQASMMakeGatesOpaquePass() {
  return std::make_unique<GatePass>();
}
} // namespace mlir
