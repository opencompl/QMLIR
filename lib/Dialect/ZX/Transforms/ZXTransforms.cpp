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

#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::ZX;

// Pattern rewriter
class ZXRewritePass : public ZXRewritePassBase<ZXRewritePass> {
  void runOnFunction() override;
};

// source: https://mlir.llvm.org/docs/PatternRewriter/

// Generic Rewrite Pattern for ZX Ops
template <typename MyOp>
class ZXRewritePattern : public RewritePattern {
public:
  ZXRewritePattern(PatternBenefit benefit, MLIRContext *context)
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
/// Rules 1, 2: Z/X Spider Fusion

template <typename NodeOp>
class ZXSpiderFusionPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    NodeOp rootNodeOp = cast<NodeOp>(op);
    Value middleWire;
    NodeOp childNodeOp;

    bool matched = false;
    for (auto input : llvm::enumerate(rootNodeOp.getInputWires())) {
      if ((childNodeOp = input.value().template getDefiningOp<NodeOp>())) {
        middleWire = input.value();
        matched = true;
        break;
      }
    }
    if (!matched)
      return failure();

    auto combinedAngle = rewriter.create<AddFOp>(
        rewriter.getUnknownLoc(), rootNodeOp.getParam().getType(),
        rootNodeOp.getParam(), childNodeOp.getParam());

    SmallVector<Value, 10> combinedInputs;
    combinedInputs.push_back(combinedAngle);
    combinedInputs.append(childNodeOp.getInputWires().begin(),
                          childNodeOp.getInputWires().end());
    for (auto input : rootNodeOp.getInputWires()) {
      if (input != middleWire) {
        combinedInputs.push_back(input);
      }
    }

    SmallVector<Type, 10> combinedOutputTypes(
        (childNodeOp.getNumResults() + rootNodeOp.getNumResults()) - 1,
        rewriter.getType<ZX::WireType>());

    NodeOp combinedNodeOp = rewriter.create<NodeOp>(
        rewriter.getUnknownLoc(), combinedOutputTypes, combinedInputs);
    auto combinedOutputs = combinedNodeOp.getResults();

    SourceNodeOp dummySource = rewriter.create<SourceNodeOp>(
        rewriter.getUnknownLoc(), TypeRange{rewriter.getType<ZX::WireType>()},
        ValueRange{});
    SmallVector<Value, 10> newChildNodeResults{combinedOutputs.begin() +
                                                   rootNodeOp.getNumResults(),
                                               combinedOutputs.end()};
    newChildNodeResults.append(dummySource.getResults().begin(),
                               dummySource.getResults().end());

    rewriter.replaceOp(rootNodeOp,
                       combinedOutputs.take_front(rootNodeOp.getNumResults()));
    rewriter.replaceOp(childNodeOp, newChildNodeResults);

    return success();
  }
};

/// Populate the pattern list.
void collectZXRewritePatterns(OwningRewritePatternList &patterns,
                              MLIRContext *ctx) {
  patterns.insert<ZXSpiderFusionPattern<ZOp>>(1, ctx);
  patterns.insert<ZXSpiderFusionPattern<XOp>>(1, ctx);
}

void ZXRewritePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns;
  collectZXRewritePatterns(patterns, &getContext());

  applyPatternsAndFoldGreedily(func, std::move(patterns));
}

namespace mlir {

std::unique_ptr<FunctionPass> createTransformZXRewritePass() {
  return std::make_unique<ZXRewritePass>();
}

} // namespace mlir
