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

class QuantumRewritePass : public QuantumRewritePassBase<QuantumRewritePass> {
  void runOnFunction() override;
};

namespace {
#include "Dialect/Quantum/Transforms/QuantumRewrites.h.inc"
} // namespace

struct EulerAngles {
  double theta, phi, lambd;
  static EulerAngles fromYZY(double theta, double phi, double lambd) {
    /// Dummy implementation for testing
    /// TODO: implement actual algorithm
    return {theta, phi, lambd};
  }
};

// U(theta, phi, lambda)
//           = U(theta2, phi2, lambda2).U(theta1, phi1, lambda1)
//           = Rz(phi2).Ry(theta2).Rz(lambda2+phi1).Ry(theta1).Rz(lambda1)
//           = Rz(phi2).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda1)
//           = U(theta', phi2 + phi', lambda1 + lambda')
class UOpMergePattern : public OpRewritePattern<UniversalRotationGateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UniversalRotationGateOp uOp,
                                PatternRewriter &rewriter) const final {
    auto parentUOp = uOp.qinp().getDefiningOp<UniversalRotationGateOp>();
    if (!parentUOp)
      return failure();

    /// Dummy implementation for testing only
    /// TODO: implement proper merge: U.U
    rewriter.replaceOp(uOp, parentUOp.qout());

    return success();
  }
};

// %b1, %a1 = CNOT %b0, %a0
// %a2, %b2 = CNOT %a1, %b1
// -----
// %b2, %a2 = CNOT %a0, %b0
struct AlternateCNOTPattern : public OpRewritePattern<CNOTGateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CNOTGateOp op,
                                PatternRewriter &rewriter) const final {
    auto parentOp = op.qinp_cont().getDefiningOp<CNOTGateOp>();
    if (!parentOp || parentOp != op.qinp_targ().getDefiningOp<CNOTGateOp>())
      return failure();
    if (op.qinp_cont() != parentOp.qout_targ())
      return failure();
    if (op.qinp_targ() != parentOp.qout_cont())
      return failure();

    auto results = rewriter.create<CNOTGateOp>(
        op->getLoc(), parentOp.qinp_targ(), parentOp.qinp_cont());
    rewriter.replaceOp(op, {results.qout_targ(), results.qout_cont()});
    return success();
  }
};

void QuantumRewritePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.insert<UOpMergePattern, AlternateCNOTPattern>(&getContext());
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
