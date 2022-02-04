#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"

using namespace mlir;
using namespace mlir::quantum;

class DepthComputePass : public QuantumDepthComputePassBase<DepthComputePass> {
  void runOnFunction() override;
};
template <class Op>
int64_t getMaxDepthOfArguments(Op op) {
  int64_t depth = 0;
  for (auto operand : op->getOperands()) {
    if (operand.getType().template isa<QubitType>()) {
      if (auto parentOp = operand.getDefiningOp()) {
        if (auto depthAttr =
                parentOp->template getAttrOfType<IntegerAttr>("qdepth")) {
          depth = std::max(depth, depthAttr.getInt());
        }
      }
    }
  }
  return depth;
}

template <class Op, int Contrib>
struct DepthComputePattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    if (op->hasAttr("qdepth")) {
      return failure();
    }
    int64_t depth = getMaxDepthOfArguments(op) + Contrib;
    auto depthAttrType = IntegerType::get(rewriter.getContext(), 64);
    auto depthAttr = IntegerAttr::get(depthAttrType, depth);
    rewriter.updateRootInPlace(op, [&]() { op->setAttr("qdepth", depthAttr); });
    return success();
  }
};

struct FunctionDepthComputePattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match(ReturnOp op) const final {
    return success(!op->hasAttr("qdepth"));
  }
  void rewrite(ReturnOp op, PatternRewriter &rewriter) const final {}
};

struct SCFIfDepthComputePattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match(scf::IfOp op) const final {
    bool hasQubits = false;
    for (auto ty : op.getResultTypes()) {
      if (ty.isa<quantum::QubitType>())
        hasQubits = true;
    }
    if (!hasQubits)
      return failure();
    return success(!op->hasAttr("qdepth"));
  }
  void rewrite(scf::IfOp op, PatternRewriter &rewriter) const final {
    int64_t depth = 0;
    for (auto yield : op.thenRegion().getOps<scf::YieldOp>()) {
      depth = std::max(depth, getMaxDepthOfArguments(&yield));
    }
    for (auto yield : op.elseRegion().getOps<scf::YieldOp>()) {
      depth = std::max(depth, getMaxDepthOfArguments(&yield));
    }

    auto depthAttrType = IntegerType::get(rewriter.getContext(), 64);
    auto depthAttr = IntegerAttr::get(depthAttrType, depth);
    rewriter.updateRootInPlace(op, [&]() { op->setAttr("qdepth", depthAttr); });
  }
};

void DepthComputePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<
      // clang-format off
      // allocate
      DepthComputePattern<AllocateOp, 0>,
      // gate
      DepthComputePattern<PauliXGateOp, 1>,
      DepthComputePattern<PauliYGateOp, 1>,
      DepthComputePattern<PauliZGateOp, 1>,
      DepthComputePattern<HadamardGateOp, 1>,
      DepthComputePattern<PhaseGateOp, 1>,
      DepthComputePattern<PhaseDaggerGateOp, 1>,
      DepthComputePattern<TGateOp, 1>,
      DepthComputePattern<TDaggerGateOp, 1>,
      DepthComputePattern<RotateXOp, 1>,
      DepthComputePattern<RotateYOp, 1>,
      DepthComputePattern<RotateZOp, 1>,
      DepthComputePattern<CNOTGateOp, 1>,
      DepthComputePattern<UniversalRotationGateOp, 1>,
      // measure
      DepthComputePattern<MeasureQubitOp, 0>,
      DepthComputePattern<MeasureOp, 0>,
      DepthComputePattern<SinkOp, 0>,
      // manipulate
      DepthComputePattern<DimensionOp, 0>,
      DepthComputePattern<CastOp, 0>,
      DepthComputePattern<ConcatOp, 0>,
      DepthComputePattern<SplitOp, 0>,
      DepthComputePattern<BarrierOp, 0>,
      // control flow
      DepthComputePattern<scf::YieldOp, 0>,
      SCFIfDepthComputePattern
      // clang-format on
      >(&getContext());
  const FrozenRewritePatternSet frozenPatterns = std::move(patterns);
  getFunction().walk([&](Operation *op) {
    if (isa<FuncOp>(op))
      return;
    applyOpPatternsAndFold(op, frozenPatterns);
  });
}

class DepthClearPass : public QuantumDepthClearPassBase<DepthClearPass> {
  void runOnFunction() override;
};

struct ClearDepthPattern : public RewritePattern {
  ClearDepthPattern(MLIRContext *ctx)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), 1, ctx) {}
  LogicalResult match(Operation *op) const final {
    return success(op->hasAttr("qdepth"));
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&]() { op->removeAttr("qdepth"); });
  }
};

void DepthClearPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<ClearDepthPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                          false))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumDepthComputePass() {
  return std::make_unique<DepthComputePass>();
}
std::unique_ptr<FunctionPass> createQuantumClearDepthPass() {
  return std::make_unique<DepthClearPass>();
}

} // namespace mlir
