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

struct QASMTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;
  QASMTypeConverter(MLIRContext *context) : context(context) {
    addConversion([&](QASM::QubitType qubitType) -> Type {
      return quantum::QubitType::get(qubitType.getContext(), 1);
    });
    addConversion([&](Type type) -> Optional<Type> {
      if (type.isa<QASM::QubitType>())
        return llvm::None;
      return type;
    });
  }
  MLIRContext *getContext() { return context; }

private:
  MLIRContext *context;
};

// Base Pattern
template <typename SourceOp>
class QASMOpToQuantumConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  QASMOpToQuantumConversionPattern(QASMTypeConverter typeConverter,
                                   PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, typeConverter.getContext(),
                                      benefit) {}
};

//====== PATTERNS ======
class PIOpLowering : public QASMOpToQuantumConversionPattern<QASM::PIOp> {
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

void populateQASMToQuantumConversionPatterns(
    QASMTypeConverter &typeConverter, OwningRewritePatternList &patterns) {
  patterns.insert<PIOpLowering>(typeConverter);
}

struct QASMToQuantumPass : public QASMToQuantumPassBase<QASMToQuantumPass> {
  void runOnOperation() override;
};

struct QASMToQuantumTarget : public ConversionTarget {
  QASMToQuantumTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<quantum::QuantumDialect>();

    addIllegalDialect<QASM::QASMDialect>();
    addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) -> bool { return true; });
  }
  // bool isDynamicallyLegal(Operation *op) const override { return true; }
};

void QASMToQuantumPass::runOnOperation() {
  OwningRewritePatternList patterns;
  QASMTypeConverter typeConverter(&getContext());
  populateQASMToQuantumConversionPatterns(typeConverter, patterns);

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
