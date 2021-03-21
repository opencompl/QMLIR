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
  [[maybe_unused]] MLIRContext *ctx;
  llvm::StringMap<DenseMap<Value, Value>> mapping;

public:
  QubitMap(MLIRContext *ctx) : ctx(ctx), mapping() {}
  void allocateQubit(FuncOp func, Value qubit) {
    mapping[func.getName()][qubit] = qubit;
  }
  Value resolveQubit(FuncOp func, Value qubit) {
    return mapping[func.getName()][qubit];
  }
  void updateQubit(FuncOp func, Value base, Value current) {
    mapping[func.getName()][base] = current;
  }
};

class QASMTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QASMTypeConverter(MLIRContext *context) : context(context) {
    addConversion([&](Type type) -> Optional<Type> {
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
  QASMOpToQuantumConversionPattern(QASMTypeConverter &typeConverter,
                                   QubitMap *qubitMap,
                                   PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, typeConverter.getContext(),
                                      benefit),
        qubitMap(qubitMap), ctx(typeConverter.getContext()) {}
};

//====== PATTERNS ======
// qasm.pi : f*
// [[to]]
// constant [M_PI] : f*
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

// %q = qasm.allocate
// [[to]]
// %q = qssa.allocate() : !qssa.qubit<1>
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
    qubitMap->allocateQubit(op->getParentOfType<FuncOp>(), allocOp.getResult());
    return success();
  }
};

// qasm.U(%theta : f*, %phi : f*, %lambda : f*) %q
// [[to]]
// %q_{i} =
//    qssa.U(%theta : f*, %phi : f*, %lambda : f*) %q_{i - 1} : !qssa.qubit<1>
class SingleQubitRotationOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::SingleQubitRotationOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::SingleQubitRotationOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};

// qasm.CX %a, %b
// [[to]]
// %a_{i}, %b_{j} = qssa.CNOT %a_{i - 1} %b_{j - 1}
class ControlledNotOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::ControlledNotOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::ControlledNotOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::ControlledNotOpAdaptor args(operands);
    auto func = op->getParentOfType<FuncOp>();
    auto cont = qubitMap->resolveQubit(func, args.qinp0());
    auto targ = qubitMap->resolveQubit(func, args.qinp1());
    auto convertedOp = rewriter.create<quantum::CNOTGateOp>(
        rewriter.getUnknownLoc(), cont, targ);
    rewriter.eraseOp(op);
    qubitMap->updateQubit(func, args.qinp0(), convertedOp.qout_cont());
    qubitMap->updateQubit(func, args.qinp1(), convertedOp.qout_targ());
    return success();
  }
};

void populateQASMToQuantumConversionPatterns(
    QASMTypeConverter &typeConverter, QubitMap &qubitMap,
    OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
    PIOpConversion,
    AllocateOpConversion,
    ControlledNotOpConversion
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
