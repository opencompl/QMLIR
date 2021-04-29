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
#include "llvm/ADT/TypeSwitch.h"

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
struct ZXRewritePattern : public OpRewritePattern<MyOp> {
  /// Helpers

  /// Add two floating point angles modulo (2*pi) and return the sum
  Value addAngles(PatternRewriter &rewriter, Value a, Value b) const {
    assert(a.getType().isa<FloatType>() && "Angle not float!!!");
    auto angleSum =
        rewriter.create<AddFOp>(rewriter.getUnknownLoc(), a.getType(), a, b);
    auto twoPI = rewriter.create<ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getFloatAttr(a.getType(), 2 * M_PI));
    auto angleSumModTwoPi = rewriter.create<RemFOp>(
        rewriter.getUnknownLoc(), angleSum.getResult(), twoPI.getResult());
    return angleSumModTwoPi;
  }

  /// Check if a floating angle value is zero or not
  bool isAngleZero(Value a) const {
    assert(a.getType().isa<FloatType>() && "Angle not float!!!");
    if (auto constOp = a.getDefiningOp<ConstantFloatOp>()) {
      return constOp.getValue().isZero();
    }
    return false;
  }

public:
  using OpRewritePattern<MyOp>::OpRewritePattern;
};

//=== Support Rules: Simplifications ===//
template <typename NodeOp>
struct RemoveDeadWirePattern : public ZXRewritePattern<NodeOp> {
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(NodeOp nodeOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> results;
    bool hasDeadWires = false;
    for (Value wire : nodeOp.outputWires()) {
      if (!wire.getUses().empty()) {
        results.push_back(wire);
        resultTypes.push_back(wire.getType());
      } else {
        hasDeadWires = true;
      }
    }
    if (!hasDeadWires)
      return failure();
    auto newNodeOp = rewriter.create<NodeOp>(
        nodeOp->getLoc(), resultTypes, nodeOp.param(), nodeOp.inputWires());
    for (auto it : llvm::zip(results, newNodeOp.outputWires())) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
    }
    rewriter.eraseOp(nodeOp);
    return success();
  }
};

//======================== ZX Rewrite Rules ===============================//

/// Rules 1, 2: Z/X Spider Fusion
template <typename NodeOp>
struct ZXSpiderFusionPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(NodeOp rootNodeOp,
                                PatternRewriter &rewriter) const override {
    Value middleWire;
    NodeOp childNodeOp;

    bool matched = false;
    for (auto input : llvm::enumerate(rootNodeOp.inputWires())) {
      if ((childNodeOp = input.value().template getDefiningOp<NodeOp>())) {
        middleWire = input.value();
        matched = true;
        if (childNodeOp.param().getType() != rootNodeOp.param().getType())
          matched = false;
        break;
      }
    }

    if (!matched)
      return failure();

    Value combinedAngle = ZXRewritePattern<NodeOp>::addAngles(
        rewriter, childNodeOp.param(), rootNodeOp.param());

    /// angle, childNodeInputs..., rootNodeInputs...
    SmallVector<Value, 5> combinedInputs;
    combinedInputs.push_back(combinedAngle);
    combinedInputs.append(childNodeOp.inputWires().begin(),
                          childNodeOp.inputWires().end());
    for (auto input : rootNodeOp.inputWires()) {
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

/// Identity Rule
/// --Z(0)-- = ----- = --X(0)--
template <typename NodeOp>
struct ZXIdentityPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;
  using ZXRewritePattern<NodeOp>::isAngleZero;

  LogicalResult match(NodeOp nodeOp) const final {
    return success(isAngleZero(nodeOp.param()) &&
                   nodeOp.inputWires().size() == 1 &&
                   nodeOp.outputWires().size() == 1);
  }

  void rewrite(NodeOp nodeOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOp(nodeOp, nodeOp.inputWires());
  }
};

template <typename NodeOp>
struct ZXIdentityInputOrOutputPattern : public ZXRewritePattern<NodeOp> {
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;
  using ZXRewritePattern<NodeOp>::isAngleZero;

  LogicalResult matchAndRewrite(NodeOp nodeOp,
                                PatternRewriter &rewriter) const final {
    if (!isAngleZero(nodeOp.param()))
      return failure();

    // auto numInputs = nodeOp.inputWires().size();
    // auto numOutputs = nodeOp.outputWires().size();
    // if ((numInputs == 2 && numOutputs == 0) ||
    //     (numInputs == 0 && numOutputs == 2)) {
    //   rewriter.eraseOp(nodeOp);
    //   return success();
    // }
    return failure();
  }
};

namespace {
#include "Dialect/ZX/Transforms/ZXRewrites.h.inc"
} // namespace

/// Populate the pattern list.
void collectZXRewritePatterns(OwningRewritePatternList &patterns,
                              MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      RemoveDeadWirePattern<ZOp>,
      RemoveDeadWirePattern<XOp>
  >(ctx, 10);
  patterns.insert<
      ZXSpiderFusionPattern<ZOp>,
      ZXSpiderFusionPattern<XOp>
  >(ctx, 1);
  patterns.insert<
      ZXIdentityPattern<ZOp>,
      ZXIdentityPattern<XOp>,
      ZXIdentityInputOrOutputPattern<ZOp>,
      ZXIdentityInputOrOutputPattern<XOp>
  >(ctx, 1);
  // clang-format on

  populateWithGenerated(patterns);
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
