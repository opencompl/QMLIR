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
  if (!::llvm::DebugFlag)
    return;
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
//=========================================================================//
// ZX Graph Rewrites
//=========================================================================//
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

  LogicalResult checkCNOT(WireOp wireOp) const {
    if (wireOp.lhs().getDefiningOp<ZNodeOp>() &&
        wireOp.rhs().getDefiningOp<XNodeOp>())
      return success();
    if (wireOp.rhs().getDefiningOp<ZNodeOp>() &&
        wireOp.lhs().getDefiningOp<XNodeOp>())
      return success();
    return failure();
  }

  SmallVector<WireOp> getHadamardInputs(HNodeOp hadamard,
                                        WireOp excludeWire) const {
    SmallVector<WireOp> v;
    for (auto *use : hadamard->getUsers()) {
      if (auto wire = dyn_cast<WireOp>(use)) {
        if (wire != excludeWire)
          v.push_back(wire);
      }
    }
    return v;
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

    // FIXME: select insertion point properly
    for (auto &inst : *op->getBlock()) {
      if (!isa<NodeOp>(inst))
        continue;
      auto curOp = dyn_cast<NodeOp>(inst);
      if (curOp == lhsOp || curOp == rhsOp)
        rewriter.setInsertionPointAfterValue(curOp.getResult());
    }

    // rewriter.setInsertionPointAfterValue(lhsOp.getResult());
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

class MultiEdgeElimination : public ZXGraphRewritePattern<WireOp> {
public:
  using ZXGraphRewritePattern::ZXGraphRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    WireOp wireOp = cast<WireOp>(op);
    if (failed(checkCNOT(wireOp)))
      return failure();

    SmallVector<WireOp, 10> parallelWires;
    for (auto *useOp : wireOp.lhs().getUsers()) {
      if (auto currentWire = dyn_cast<WireOp>(useOp)) {
        if (currentWire == wireOp)
          continue;
        if (currentWire.getOtherOperand(wireOp.lhs()) == wireOp.rhs()) {
          parallelWires.push_back(currentWire);
        }
      }
    }

    if (parallelWires.empty())
      return failure();

    debugBeforeRewrite("MULTIEDGE ELIMINATION", op);

    if (parallelWires.size() % 2 == 0)
      parallelWires.pop_back();
    rewriter.eraseOp(wireOp);
    for (auto wire : parallelWires)
      rewriter.eraseOp(wire);
    return success();
  }
};

class DoubleHadamardChainElimination : public ZXGraphRewritePattern<WireOp> {
public:
  using ZXGraphRewritePattern::ZXGraphRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    WireOp wireOp = cast<WireOp>(op);
    HNodeOp lhsOp, rhsOp;
    if (!(lhsOp = wireOp.lhs().getDefiningOp<HNodeOp>()))
      return failure();
    if (!(rhsOp = wireOp.rhs().getDefiningOp<HNodeOp>()))
      return failure();

    auto lhsInputs = getHadamardInputs(lhsOp, wireOp);
    auto rhsInputs = getHadamardInputs(rhsOp, wireOp);

    if (lhsInputs.size() != 1 || rhsInputs.size() != 1)
      return failure();

    debugBeforeRewrite("HADAMARD CHAIN ELIMINATION", op);

    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    rewriter.create<WireOp>(rewriter.getUnknownLoc(),
                            lhsInputs[0].getOtherOperand(wireOp.lhs()),
                            rhsInputs[0].getOtherOperand(wireOp.rhs()));
    rewriter.eraseOp(wireOp);
    rewriter.eraseOp(lhsInputs[0]);
    rewriter.eraseOp(rhsInputs[0]);
    rewriter.eraseOp(lhsOp);
    rewriter.eraseOp(rhsOp);

    return success();
  }
};

template <typename NodeOp, typename OtherNodeOp, int Benefit>
class ColorChange : public ZXGraphRewritePattern<NodeOp> {
  int computeBenefit(int hadamard, int total) const {
    int nonHadamard = total - hadamard;
    return hadamard - nonHadamard;
  }
  bool verifyBenefit(int hadamard, int total) const {
    return computeBenefit(hadamard, total) >= Benefit;
  }

public:
  using ZXGraphRewritePattern<NodeOp>::ZXGraphRewritePattern;
  using ZXGraphRewritePattern<NodeOp>::getHadamardInputs;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    NodeOp nodeOp = cast<NodeOp>(op);

    SmallVector<WireOp> hadamardWires, nonHadamardWires, allWires;
    for (auto *inst : nodeOp->getUsers()) {
      if (auto wire = dyn_cast<WireOp>(inst)) {
        if (wire.getOtherOperand(nodeOp.getResult())
                .template getDefiningOp<HNodeOp>()) {
          hadamardWires.push_back(wire);
        } else {
          nonHadamardWires.push_back(wire);
        }
        allWires.push_back(wire);
      }
    }
    if (!verifyBenefit(hadamardWires.size(), allWires.size()))
      return failure();

    debugBeforeRewrite("COLOR CHANGE (H switch)", op);

    for (auto wire : nonHadamardWires) {
      Value otherNode = wire.getOtherOperand(nodeOp);

      rewriter.setInsertionPointToStart(op->getBlock());
      auto middleHNode = rewriter.create<HNodeOp>(rewriter.getUnknownLoc());

      rewriter.setInsertionPoint(op->getBlock()->getTerminator());
      rewriter.create<WireOp>(rewriter.getUnknownLoc(), nodeOp, middleHNode);
      rewriter.create<WireOp>(rewriter.getUnknownLoc(), middleHNode, otherNode);

      rewriter.eraseOp(wire);
    }

    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    for (auto wire : hadamardWires) {
      Value otherNode = wire.getOtherOperand(nodeOp);
      auto hWires = getHadamardInputs(otherNode.getDefiningOp<HNodeOp>(), wire);
      for (auto otherWire : hWires) {
        Value thirdNode = otherWire.getOtherOperand(otherNode);
        rewriter.eraseOp(wire);
        rewriter.eraseOp(otherWire);
        rewriter.eraseOp(otherNode.getDefiningOp<HNodeOp>());
        rewriter.create<WireOp>(rewriter.getUnknownLoc(), nodeOp, thirdNode);
      }
    }

    rewriter.setInsertionPointAfter(op);
    auto newNode =
        rewriter.create<OtherNodeOp>(rewriter.getUnknownLoc(), nodeOp.param());
    rewriter.replaceOp(op, newNode.getResult());

    return success();
  }
};

template <typename NodeOp, typename OtherNodeOp>
class ReplicateNode : public ZXGraphRewritePattern<NodeOp> {
public:
  using ZXGraphRewritePattern<NodeOp>::ZXGraphRewritePattern;
  using ZXGraphRewritePattern<NodeOp>::checkZero;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    NodeOp nodeOp = cast<NodeOp>(op);
    if (!nodeOp->hasOneUse())
      return failure();
    if (!checkZero(nodeOp.param()))
      return failure();

    WireOp wireOp;
    if (!(wireOp = dyn_cast<WireOp>(*nodeOp->getUsers().begin())))
      return failure();
    OtherNodeOp otherNodeOp;
    if (!(otherNodeOp = wireOp.getOtherOperand(nodeOp)
                            .template getDefiningOp<OtherNodeOp>()))
      return failure();

    for (auto inst : otherNodeOp->getUsers()) {
      if (WireOp nextWire = dyn_cast<WireOp>(inst)) {
        if (nextWire == wireOp)
          continue;
        auto nextNode = nextWire.getOtherOperand(otherNodeOp);
        rewriter.setInsertionPointAfter(op);
        auto *nodeClone = rewriter.clone(*op);
        rewriter.setInsertionPoint(op->getBlock()->getTerminator());
        rewriter.create<WireOp>(rewriter.getUnknownLoc(),
                                nodeClone->getResult(0), nextNode);

        rewriter.eraseOp(nextWire);
      }
    }
    rewriter.eraseOp(wireOp);
    rewriter.eraseOp(op);
    rewriter.eraseOp(otherNodeOp);

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

  patterns.insert<SpiderFusion<ZNodeOp>>(10, ctx);
  patterns.insert<SpiderFusion<XNodeOp>>(10, ctx);

  patterns.insert<IdentityRule<ZNodeOp>>(10, ctx);
  patterns.insert<IdentityRule<XNodeOp>>(10, ctx);

  patterns.insert<MultiEdgeElimination>(10, ctx);
  patterns.insert<MultiEdgeElimination>(10, ctx);

  patterns.insert<DoubleHadamardChainElimination>(10, ctx);

  patterns.insert<ReplicateNode<ZNodeOp, XNodeOp>>(10, ctx);
  patterns.insert<ReplicateNode<XNodeOp, ZNodeOp>>(10, ctx);

  patterns.insert<ColorChange<ZNodeOp, XNodeOp, 1>>(1, ctx);
  patterns.insert<ColorChange<ZNodeOp, XNodeOp, 2>>(2, ctx);
  patterns.insert<ColorChange<ZNodeOp, XNodeOp, 3>>(3, ctx);
  patterns.insert<ColorChange<ZNodeOp, XNodeOp, 4>>(4, ctx);

  patterns.insert<ColorChange<XNodeOp, ZNodeOp, 1>>(1, ctx);
  patterns.insert<ColorChange<XNodeOp, ZNodeOp, 2>>(2, ctx);
  patterns.insert<ColorChange<XNodeOp, ZNodeOp, 3>>(3, ctx);
  patterns.insert<ColorChange<XNodeOp, ZNodeOp, 4>>(4, ctx);

  WireOp::getCanonicalizationPatterns(patterns, ctx);
}

// Pattern rewriter
class ZXGraphRewritePass : public ZXGraphRewritePassBase<ZXGraphRewritePass> {
  void runOnFunction() override;
};

void ZXGraphRewritePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns(&getContext());
  collectZXGraphRewritePatterns(patterns, &getContext());

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
  }
}

} // namespace

//===========================================================================//
// ZX Block Canonicalizer
//===========================================================================//
namespace {

class ZXWireCanonicalizerPattern : public ZXGraphRewritePattern<WireOp> {
public:
  using ZXGraphRewritePattern::ZXGraphRewritePattern;
  LogicalResult match(Operation *op) const override {
    auto wireOp = cast<WireOp>(op);
    bool opSeen = false;
    for (auto &inst : *op->getBlock()) {
      if (isa<WireOp>(inst)) {
        auto currentWireOp = cast<WireOp>(inst);
        if (currentWireOp == wireOp)
          opSeen = true;
      } else {
        if (inst.hasTrait<OpTrait::IsTerminator>())
          continue;
        if (opSeen)
          return success();
      }
    }
    return failure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto wireOp = cast<WireOp>(op);
    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    rewriter.create<WireOp>(rewriter.getUnknownLoc(), wireOp.lhs(),
                            wireOp.rhs());
    rewriter.eraseOp(op);
  }
};

/// Populate the pattern list.
static void
collectZXGraphCanonicalizerPatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *ctx) {
  WireOp::getCanonicalizationPatterns(patterns, ctx);
  patterns.insert<ZXWireCanonicalizerPattern>(1, ctx);
}

// Pattern rewriter
class ZXGraphCanonicalizePass
    : public ZXGraphCanonicalizePassBase<ZXGraphCanonicalizePass> {
  void runOnFunction() override;
};

void ZXGraphCanonicalizePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns(&getContext());
  collectZXGraphCanonicalizerPatterns(patterns, &getContext());
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
    ;
}
} // namespace

namespace mlir {

std::unique_ptr<FunctionPass> createTransformZXGraphRewritePass() {
  return std::make_unique<ZXGraphRewritePass>();
}

std::unique_ptr<FunctionPass> createTransformZXGraphCanonicalizePass() {
  return std::make_unique<ZXGraphCanonicalizePass>();
}

} // namespace mlir
