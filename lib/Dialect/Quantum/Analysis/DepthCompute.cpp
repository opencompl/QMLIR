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

class DepthComputePass : public QuantumDepthComputePassBase<DepthComputePass> {
  void runOnFunction() override;
};

template <class Op, int Contrib>
struct DepthComputePattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult match(Op op) const final {
    return success(!op->hasAttr("qdepth"));
  }
  void rewrite(Op op, PatternRewriter &rewriter) const final {
    auto depthAttrType = IntegerType::get(rewriter.getContext(), 64);
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
    depth += Contrib;
    auto depthAttr = IntegerAttr::get(depthAttrType, depth);
    rewriter.updateRootInPlace(op, [&]() { op->setAttr("qdepth", depthAttr); });
  }
};

struct FunctionDepthComputePattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match(ReturnOp op) const final {
    return success(!op->hasAttr("qdepth"));
  }
  void rewrite(ReturnOp op, PatternRewriter &rewriter) const final {}
};

void DepthComputePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<
      // clang-format off
      // allocate
      DepthComputePattern<AllocateOp, 0>,
      // gate
      DepthComputePattern<IDGateOp, 1>,
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
      DepthComputePattern<BarrierOp, 0>
      // clang-format on
      >(&getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

class DepthClearPass : public QuantumDepthClearPassBase<DepthClearPass> {
  void runOnFunction() override;
};

struct ClearDepthPattern : public RewritePattern {
  ClearDepthPattern() : RewritePattern(1, Pattern::MatchAnyOpTypeTag()) {}
  LogicalResult match(Operation *op) const final {
    return success(op->hasAttr("qdepth"));
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&]() { op->removeAttr("qdepth"); });
  }
};

void DepthClearPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<ClearDepthPattern>();
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
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
