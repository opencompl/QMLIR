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
        if (childNodeOp.getParam().getType() != rootNodeOp.getParam().getType())
          matched = false;
        break;
      }
    }

    if (!matched)
      return failure();

    Value combinedAngle = ZXRewritePattern<NodeOp>::addAngles(
        rewriter, childNodeOp.getParam(), rootNodeOp.getParam());

    /// angle, childNodeInputs..., rootNodeInputs...
    SmallVector<Value, 5> combinedInputs;
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
    /// childNodeOutputs..., rootNodeOutputs...
    ResultRange combinedOutputs = combinedNodeOp.getResults();

    auto outputIt = combinedOutputs.begin();
    for (Value output : childNodeOp.getResults()) {
      if (output != middleWire) {
        output.replaceAllUsesWith(*outputIt);
        ++outputIt;
      }
    }
    for (Value output : rootNodeOp.getResults()) {
      output.replaceAllUsesWith(*outputIt);
      ++outputIt;
    }
    rewriter.eraseOp(rootNodeOp);
    rewriter.eraseOp(childNodeOp);

    return success();
  }
};

/// Hadamard Color Change
/// --H--Z--H-- = ----X----
template <typename NodeOp, typename NewNodeOp>
class ZXHadamardColorChangePattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement.
    return failure();
    NodeOp rootNodeOp = cast<NodeOp>(op);

    bool matched = false;
    for (auto input : rootNodeOp.getInputWires()) {
      if (!input.template getDefiningOp<HOp>()) {
        matched = false;
        break;
      }
    }
    Value r;
    r.getUsers();
    for (auto output : rootNodeOp.getResults()) {
      (void)output;
    }
    if (!matched)
      return failure();

    // auto combinedAngle = rewriter.create<AddFOp>(
    //     rewriter.getUnknownLoc(), rootNodeOp.getParam().getType(),
    //     childNodeOp.getParam(), rootNodeOp.getParam());

    // /// angle, childNodeInputs..., rootNodeInputs...
    // SmallVector<Value, 10> combinedInputs;
    // combinedInputs.push_back(combinedAngle);
    // combinedInputs.append(childNodeOp.getInputWires().begin(),
    //                       childNodeOp.getInputWires().end());
    // for (auto input : rootNodeOp.getInputWires()) {
    //   if (input != middleWire) {
    //     combinedInputs.push_back(input);
    //   }
    // }

    // SmallVector<Type, 10> combinedOutputTypes(
    //     (childNodeOp.getNumResults() + rootNodeOp.getNumResults()) - 1,
    //     rewriter.getType<ZX::WireType>());

    // NodeOp combinedNodeOp = rewriter.create<NodeOp>(
    //     rewriter.getUnknownLoc(), combinedOutputTypes, combinedInputs);
    // /// childNodeOutputs..., rootNodeOutputs...
    // ResultRange combinedOutputs = combinedNodeOp.getResults();

    // auto outputIt = combinedOutputs.begin();
    // for (Value output : childNodeOp.getResults()) {
    //   if (output != middleWire) {
    //     output.replaceAllUsesWith(*outputIt);
    //     ++outputIt;
    //   }
    // }
    // for (Value output : rootNodeOp.getResults()) {
    //   output.replaceAllUsesWith(*outputIt);
    //   ++outputIt;
    // }
    // rewriter.eraseOp(rootNodeOp);
    // rewriter.eraseOp(childNodeOp);

    return success();
  }
};

/// Identity Rule
/// --Z(0)-- = ----- = --X(0)--
template <typename NodeOp>
class ZXIdentityPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult match(Operation *op) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    Value param = nodeOp.getParam();
    if (ConstantFloatOp paramOp = param.getDefiningOp<ConstantFloatOp>()) {
      if (paramOp.getValue().isZero() && nodeOp.getInputWires().size() == 1 &&
          nodeOp.getResults().size() == 1)
        return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    Value input = *nodeOp.getInputWires().begin();
    Value output = *nodeOp.getResults().begin();
    output.replaceAllUsesWith(input);
    rewriter.eraseOp(nodeOp);
  }
};

template <typename NodeOp>
class ZXIdentityResultPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult match(Operation *op) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    Value param = nodeOp.getParam();
    if (ConstantFloatOp paramOp = param.getDefiningOp<ConstantFloatOp>()) {
      if (paramOp.getValue().isZero() && nodeOp.getInputWires().size() == 0 &&
          nodeOp.getResults().size() == 2)
        return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    Value input = *nodeOp.getInputWires().begin();
    Value output = *nodeOp.getResults().begin();
    output.replaceAllUsesWith(input);
    rewriter.eraseOp(nodeOp);
  }
};

//====== Auxillary Rewrites ================================================//
class RemFRewrite : public ZXRewritePattern<RemFOp> {
public:
  using ZXRewritePattern<RemFOp>::ZXRewritePattern;

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
void collectZXRewritePatterns(OwningRewritePatternList &patterns,
                              MLIRContext *ctx) {
  patterns.insert<ZXSpiderFusionPattern<ZOp>>(1, ctx);
  patterns.insert<ZXSpiderFusionPattern<XOp>>(1, ctx);
  // patterns.insert<ZXHadamardColorChangePattern<ZOp, XOp>>(1, ctx);
  // patterns.insert<ZXHadamardColorChangePattern<XOp, ZOp>>(1, ctx);
  patterns.insert<ZXIdentityPattern<ZOp>>(1, ctx);
  patterns.insert<ZXIdentityPattern<XOp>>(1, ctx);
  patterns.insert<RemFRewrite>(1, ctx);
}

void ZXRewritePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns(&getContext());
  collectZXRewritePatterns(patterns, &getContext());

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createTransformZXRewritePass() {
  return std::make_unique<ZXRewritePass>();
}

} // namespace mlir
