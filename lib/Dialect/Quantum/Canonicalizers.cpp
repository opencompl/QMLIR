#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "Dialect/Quantum/QuantumOps.h"

using namespace mlir;
using namespace mlir::quantum;

namespace {

struct NegFRewrite : public OpRewritePattern<NegFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegFOp op,
                                PatternRewriter &rewriter) const override {
    if (auto constOp = op.operand().getDefiningOp<ConstantFloatOp>()) {
      auto value = constOp.getValue();
      APFloat m1(value.getSemantics(), "-1");
      value.multiply(m1, llvm::APFloatBase::roundingMode::TowardZero);
      rewriter.replaceOp(
          op, rewriter
                  .create<ConstantFloatOp>(op.getLoc(), value,
                                           constOp.getType().cast<FloatType>())
                  .getResult());
      return success();
    }
    return failure();
  }
};

struct RemFRewrite : public OpRewritePattern<RemFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RemFOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsOp = op.lhs().getDefiningOp<ConstantFloatOp>();
    auto rhsOp = op.rhs().getDefiningOp<ConstantFloatOp>();
    if (!lhsOp || !rhsOp)
      return failure();
    APFloat result = lhsOp.getValue();
    result.remainder(rhsOp.getValue());
    rewriter.replaceOp(
        op, rewriter
                .create<ConstantFloatOp>(op.getLoc(), result,
                                         op.getType().cast<FloatType>())
                .getResult());
    return success();
  }
};

} // namespace

// Collect all patterns
void CanonicalizeSupportOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<NegFRewrite, RemFRewrite>(context);
}