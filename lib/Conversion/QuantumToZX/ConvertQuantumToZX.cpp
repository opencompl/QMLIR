#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversion/QuantumToZX/Passes.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "PassDetail.h"

using namespace mlir;
using quantum::QubitType;

namespace {

class QuantumTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QuantumTypeConverter(MLIRContext *context) : context(context) {
    addConversion([](Type type) { return type; });
    addConversion(
        [&](QubitType type) { return ZX::WireType::get(type.getContext()); });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};
/// TODO: Implement support for multiqubit arrays

//====== PATTERNS ======
struct AllocOpConversion : public OpConversionPattern<quantum::AllocateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::AllocateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.getType().isSingleQubit()) {
      return op.emitError(
          "qssa to ZX conversion does not support multi-qubit arrays");
    }
    auto newOp = rewriter.create<ZX::SourceNodeOp>(op->getLoc());
    newOp->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct SinkOpConversion : public OpConversionPattern<quantum::SinkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::SinkOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.qinp().getType().cast<QubitType>().isSingleQubit()) {
      return op.emitError(
          "qssa to ZX conversion does not support multi-qubit arrays");
    }
    quantum::SinkOpAdaptor converted(operands);
    auto newOp =
        rewriter.create<ZX::SinkNodeOp>(op->getLoc(), converted.qinp());
    newOp->setAttrs(op->getAttrDictionary());
    rewriter.eraseOp(op);
    return success();
  }
};
struct MeasureQubitOpConversion
    : public OpConversionPattern<quantum::MeasureQubitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::MeasureQubitOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.qinp().getType().cast<QubitType>().isSingleQubit()) {
      return op.emitError(
          "qssa to ZX conversion does not support multi-qubit arrays");
    }
    quantum::MeasureQubitOpAdaptor converted(operands);
    auto newOp = rewriter.create<ZX::MeasureOp>(
        op->getLoc(), rewriter.getI1Type(), converted.qinp().getType(),
        converted.qinp());
    newOp->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct RotateXOpConversion : public OpConversionPattern<quantum::RotateXOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::RotateXOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.qinp().getType().cast<QubitType>().isSingleQubit()) {
      return op.emitError(
          "qssa to ZX conversion does not support multi-qubit arrays");
    }
    quantum::RotateXOpAdaptor converted(operands);
    auto newOp = rewriter.create<ZX::XOp>(
        op->getLoc(), TypeRange{ZX::WireType::get(getContext())},
        converted.param(), ValueRange{converted.qinp()});
    newOp->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
struct RotateZOpConversion : public OpConversionPattern<quantum::RotateZOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::RotateZOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.qinp().getType().cast<QubitType>().isSingleQubit()) {
      return op.emitError(
          "qssa to ZX conversion does not support multi-qubit arrays");
    }
    quantum::RotateZOpAdaptor converted(operands);
    auto newOp = rewriter.create<ZX::ZOp>(
        op->getLoc(), TypeRange{ZX::WireType::get(getContext())},
        converted.param(), ValueRange{converted.qinp()});
    newOp->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
struct CNOTOpConversion : public OpConversionPattern<quantum::CNOTGateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(quantum::CNOTGateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    quantum::CNOTGateOpAdaptor converted(operands);
    auto zero = getZero(rewriter);
    auto wireType = ZX::WireType::get(getContext());
    auto zOp =
        rewriter.create<ZX::ZOp>(op->getLoc(), TypeRange{wireType, wireType},
                                 zero, ValueRange{converted.qinp_cont()});
    auto xOp = rewriter.create<ZX::XOp>(
        op->getLoc(), TypeRange{wireType}, zero,
        ValueRange{zOp.getResult(1), converted.qinp_targ()});
    zOp->setAttrs(op->getAttrDictionary());
    xOp->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, {zOp.getResult(0), xOp.getResult(0)});
    return success();
  }

  Value getZero(PatternRewriter &rewriter) const {
    return rewriter.create<ConstantOp>(rewriter.getUnknownLoc(),
                                       rewriter.getF64FloatAttr(0.0));
  }
};

void populateQuantumToZXConversionPatterns(QuantumTypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      AllocOpConversion,
      RotateXOpConversion,
      RotateZOpConversion,
      CNOTOpConversion,
      SinkOpConversion,
      MeasureQubitOpConversion
  >(typeConverter, typeConverter.getContext());
  // clang-format on
}

struct QuantumToZXTarget : public ConversionTarget {
  QuantumToZXTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<ZX::ZXDialect>();
    addIllegalDialect<quantum::QuantumDialect>();
  }
};

struct QuantumToZXPass : public QuantumToZXPassBase<QuantumToZXPass> {
  void runOnOperation() override;
};

void QuantumToZXPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  QuantumTypeConverter typeConverter(&getContext());
  populateQuantumToZXConversionPatterns(typeConverter, patterns);

  QuantumToZXTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createQuantumToZXPass() {
  return std::make_unique<QuantumToZXPass>();
}
} // namespace mlir
