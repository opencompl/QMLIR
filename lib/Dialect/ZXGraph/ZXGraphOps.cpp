#include "mlir/IR/PatternMatch.h"

#include "Dialect/ZXGraph/ZXGraphOps.h"

using namespace mlir;
using namespace mlir::ZXGraph;

#define GET_OP_CLASSES
#include "Dialect/ZXGraph/ZXGraphOps.cpp.inc"

class WireOpSelfLoopRewrite : public RewritePattern {
public:
  WireOpSelfLoopRewrite(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(WireOp::getOperationName(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    WireOp wireOp = cast<WireOp>(op);
    if (wireOp.lhs() != wireOp.rhs())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

void WireOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx) {
  patterns.insert<WireOpSelfLoopRewrite>(1, ctx);
}
