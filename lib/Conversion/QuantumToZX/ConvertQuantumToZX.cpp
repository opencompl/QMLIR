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
    rewriter.create<ZX::SinkNodeOp>(op->getLoc(), converted.qinp());
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
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

void populateQuantumToZXConversionPatterns(QuantumTypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      AllocOpConversion,
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
