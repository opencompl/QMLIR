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

class GateCallRewrite : public OpRewritePattern<GateCall> {
  SmallVector<StringRef> gates;

public:
  GateCallRewrite(SmallVector<StringRef> gates, MLIRContext *ctx,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), gates(gates) {}
  LogicalResult match(GateCall op) const override {
    for (auto gate : gates) {
      if (op.gate_name() == gate)
        return success();
    }
    return failure();
  }
  void rewrite(GateCall op, PatternRewriter &rewriter) const override {
    auto callOp = rewriter.create<CallOp>(op->getLoc(), op.gate_name(),
                                          TypeRange{}, op.gate_args());
    callOp->setAttr("qasm.gate", UnitAttr::get(rewriter.getContext()));
    rewriter.eraseOp(op);
  }
};

class GatePass : public QASMMakeGatesTransparentPassBase<GatePass> {
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
std::unique_ptr<FunctionPass> createQASMMakeGatesTransparentPass() {
  return std::make_unique<GatePass>();
}
} // namespace mlir
