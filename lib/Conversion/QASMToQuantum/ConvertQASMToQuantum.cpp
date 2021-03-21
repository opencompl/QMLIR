#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversion/QASMToQuantum/Passes.h"
#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMOps.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "PassDetail.h"

using namespace mlir;

namespace {

class QubitMap {
  MLIRContext *ctx;

public:
  QubitMap(MLIRContext *ctx) : ctx(ctx) {}
};

class QASMTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QASMTypeConverter(MLIRContext *context) : context(context) {
    addConversion([&](Type type) -> Optional<Type> {
      assert(false && "reached type conversion callback");
      if (type.isa<QASM::QubitType>())
        return quantum::QubitType::get(this->context, 1);
      return Optional<Type>(type);
    });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};

// Base Pattern
template <typename SourceOp>
class QASMOpToQuantumConversionPattern : public OpConversionPattern<SourceOp> {
protected:
  QubitMap *qubitMap;
  MLIRContext *ctx;
  quantum::QubitType getSingleQubitType() const {
    return quantum::QubitType::get(ctx, 1);
  }
  Type convertType(Type type) const {
    if (type.isa<QASM::QubitType>())
      return getSingleQubitType();
    return type;
  }

public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  QASMOpToQuantumConversionPattern(QASMTypeConverter typeConverter,
                                   QubitMap *qubitMap,
                                   PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, typeConverter.getContext(),
                                      benefit),
        qubitMap(qubitMap), ctx(typeConverter.getContext()) {}
};

//====== PATTERNS ======
class PIOpConversion : public QASMOpToQuantumConversionPattern<QASM::PIOp> {
  APFloat getPIValue(Type type) const {
    if (type.isa<Float32Type>())
      return APFloat(float(M_PI));
    if (type.isa<Float64Type>())
      return APFloat(double(M_PI));
    assert(false && "invalid float type for pi");
  }

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::PIOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    APFloat pi = getPIValue(op.getType());
    auto res = rewriter.create<ConstantFloatOp>(rewriter.getUnknownLoc(), pi,
                                                op.getType().cast<FloatType>());
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

class AllocateOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::AllocateOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::AllocateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto qubitType = getSingleQubitType();
    auto allocOp = rewriter.create<quantum::AllocateOp>(
        rewriter.getUnknownLoc(), qubitType, ValueRange{});
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

void populateQASMToQuantumConversionPatterns(
    QASMTypeConverter &typeConverter, QubitMap &qubitMap,
    OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
    PIOpConversion,
    AllocateOpConversion
  >(typeConverter, &qubitMap);
  // clang-format on
}

struct QASMToQuantumTarget : public ConversionTarget {
  QASMToQuantumTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<quantum::QuantumDialect>();

    addIllegalDialect<QASM::QASMDialect>();
    addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) -> bool { return true; });
  }
  // bool isDynamicallyLegal(Operation *op) const override { return true; }
};

struct QASMToQuantumPass : public QASMToQuantumPassBase<QASMToQuantumPass> {
  void runOnOperation() override;
};

void QASMToQuantumPass::runOnOperation() {
  OwningRewritePatternList patterns;
  QASMTypeConverter typeConverter(&getContext());
  QubitMap qubitMap(&getContext());
  populateQASMToQuantumConversionPatterns(typeConverter, qubitMap, patterns);

  QASMToQuantumTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createQASMToQuantumPass() {
  return std::make_unique<QASMToQuantumPass>();
}
} // namespace mlir
