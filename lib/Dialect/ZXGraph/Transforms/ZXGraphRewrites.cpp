//===- ConvertQuantumToStandard.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/ZXGraph/ZXGraphDialect.h"
#include "Dialect/ZXGraph/ZXGraphOps.h"
#include "PassDetail.h"

#include <string>

using namespace mlir;
using namespace mlir::ZXGraph;

namespace {
// DEBUG HELPERS
static void debugBeforeRewrite(StringRef patName, Operation *op) {
  llvm::errs() << std::string(40, '>');
  llvm::errs() << "\n";
  llvm::errs() << ">>>> REWRITING: " << patName << "\n";
  op->print(llvm::errs());
  llvm::errs() << "\n\n";
  op->getBlock()->print(llvm::errs());
  llvm::errs() << std::string(40, '<');
  llvm::errs() << "\n\n";
}

} // namespace

// source: https://mlir.llvm.org/docs/PatternRewriter/

namespace {
// Generic Rewrite Pattern for ZX Ops
template <typename MyOp>
class ZXGraphRewritePattern : public RewritePattern {
  /// Helpers
protected:
  ConstantFloatOp insertConstantFloat(PatternRewriter &rewriter, APFloat v,
                                      FloatType floatType = FloatType()) const {
    // TODO: figure out the right way to do this
    return rewriter.create<ConstantFloatOp>(rewriter.getUnknownLoc(), v,
                                            floatType);
  }

  Value addAngles(PatternRewriter &rewriter, Value a, Value b) const {
    assert(a.getType().isa<FloatType>() && "Angle not float!!!");
    auto combinedAngle =
        rewriter.create<AddFOp>(rewriter.getUnknownLoc(), a.getType(), a, b);
    auto two = insertConstantFloat(rewriter, APFloat(2.0f),
                                   a.getType().cast<FloatType>());
    auto combinedAngleModuloTwo = rewriter.create<RemFOp>(
        rewriter.getUnknownLoc(), combinedAngle.getResult(), two.getResult());
    return combinedAngleModuloTwo;
  }

  bool checkZero(Value v) const {
    if (auto constOp = v.getDefiningOp<ConstantFloatOp>()) {
      return constOp.getValue().isZero();
    }
    return false;
  }

public:
  ZXGraphRewritePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  // LogicalResult match(Operation *op) const override {
  //   // The `match` method returns `success()` if the pattern is a match,
  //   // failure otherwise.
  //   // ...
  // }
  // void rewrite(Operation *op, PatternRewriter &rewriter) const override {
  //   // The `rewrite` method performs mutations on the IR rooted at `op` using
  //   // the provided rewriter. All mutations must go through the provided
  //   // rewriter.
  // }
  /// In this section, the `match` and `rewrite` implementation is specified
  /// using a single hook.
  // LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) {
  //   // The `matchAndRewrite` method performs both the matching and the
  //   // mutation.
  //   // Note that the match must reach a successful point before IR mutation
  //   // may take place.
  // }
};

//======================== ZX Rewrite Rules ===============================//

template <typename NodeOp>
class SpiderFusion : public ZXGraphRewritePattern<WireOp> {
public:
  using ZXGraphRewritePattern::ZXGraphRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    WireOp wireOp = cast<WireOp>(op);
    NodeOp lhsOp, rhsOp;
    if (!(lhsOp = wireOp.lhs().getDefiningOp<NodeOp>()))
      return failure();
    if (!(rhsOp = wireOp.rhs().getDefiningOp<NodeOp>()))
      return failure();

    debugBeforeRewrite("SPIDER FUSION", op);

    rewriter.setInsertionPointAfterValue(lhsOp.getResult());
    Value angle = addAngles(rewriter, lhsOp.param(), rhsOp.param());
    NodeOp combinedOp =
        rewriter.create<NodeOp>(rewriter.getUnknownLoc(), angle);

    rewriter.replaceOp(lhsOp, combinedOp.getResult());
    rewriter.replaceOp(rhsOp, combinedOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename NodeOp>
class IdentityRule : public ZXGraphRewritePattern<NodeOp> {
public:
  using ZXGraphRewritePattern<NodeOp>::ZXGraphRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    Value node = nodeOp.getResult();
    if (!ZXGraphRewritePattern<NodeOp>::checkZero(nodeOp.param()))
      return failure();

    auto uses = nodeOp.getResult().getUses();
    SmallVector<WireOp, 2> useWires;
    for (auto &use : uses) {
      useWires.push_back(cast<WireOp>(use.getOwner()));
    }
    if (useWires.size() != 2u)
      return failure();
    if (useWires[0] == useWires[1])
      return failure();

    debugBeforeRewrite("IDENTITY REMOVAL", op);

    SmallVector<Value, 2> otherNodes;
    for (auto wireOp : useWires) {
      for (auto otherNode : wireOp.getOperands()) {
        if (otherNode != node)
          otherNodes.push_back(otherNode);
      }
    }

    // rewriter.setInsertionPointAfter(useWires[1]);
    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    [[maybe_unused]] auto newWireOp = rewriter.create<WireOp>(
        rewriter.getUnknownLoc(), otherNodes[0], otherNodes[1]);

    for (auto wireOp : useWires) {
      rewriter.eraseOp(wireOp);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//====== Auxillary Rewrites ================================================//
class RemFRewrite : public ZXGraphRewritePattern<RemFOp> {
public:
  using ZXGraphRewritePattern<RemFOp>::ZXGraphRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    RemFOp remFOp = cast<RemFOp>(op);
    Value lhs = remFOp.lhs(), rhs = remFOp.rhs();
    ConstantFloatOp lhsOp, rhsOp;
    if (!(lhsOp = lhs.getDefiningOp<ConstantFloatOp>()))
      return failure();
    if (!(rhsOp = rhs.getDefiningOp<ConstantFloatOp>()))
      return failure();

    APFloat rem = lhsOp.getValue();
    rem.mod(rhsOp.getValue());
    auto computed =
        insertConstantFloat(rewriter, rem, lhs.getType().cast<FloatType>());

    rewriter.replaceOp(op, computed.getResult());
    return success();
  }
};

/// Populate the pattern list.
static void collectZXGraphRewritePatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *ctx) {
  patterns.insert<RemFRewrite>(1, ctx);
  patterns.insert<SpiderFusion<ZNodeOp>>(1, ctx);
  patterns.insert<SpiderFusion<XNodeOp>>(1, ctx);
  patterns.insert<IdentityRule<ZNodeOp>>(10, ctx);
  patterns.insert<IdentityRule<XNodeOp>>(10, ctx);
  WireOp::getCanonicalizationPatterns(patterns, ctx);
}

// Pattern rewriter
class ZXGraphRewritePass : public ZXGraphRewritePassBase<ZXGraphRewritePass> {
  void runOnFunction() override;
};

void ZXGraphRewritePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns;
  collectZXGraphRewritePatterns(patterns, &getContext());

  applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

namespace mlir {

std::unique_ptr<FunctionPass> createTransformZXGraphRewritePass() {
  return std::make_unique<ZXGraphRewritePass>();
}

} // namespace mlir
