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
  llvm::StringMap<SmallVector<Value>> functionArguments;

public:
  QubitMap(MLIRContext *ctx) : ctx(ctx), mapping(), functionArguments() {}

  /// add a new allocated qubit
  /// if it is a function argument, mark it, for flattening the return values
  void allocateQubit(FuncOp func, Value qubit, bool isFuncArgument = false) {
    mapping[func.getName()][qubit] = qubit;
    if (isFuncArgument)
      functionArguments[func.getName()].push_back(qubit);
  }

  /// get the current state of a qubit
  Value resolveQubit(FuncOp func, Value qubit) {
    return mapping[func.getName()][qubit];
  }

  /// update the state of a qubit, after applying an operation/function call
  void updateQubit(FuncOp func, Value base, Value current) {
    mapping[func.getName()][base] = current;
  }

  /// return the final state of the qubit arguments
  SmallVector<Value> flattenResults(FuncOp func) {
    SmallVector<Value> results;
    for (auto arg : functionArguments[func.getName()])
      results.push_back(mapping[func.getName()][arg]);
    return results;
  }
};

class QASMTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QASMTypeConverter(MLIRContext *context) : context(context) {
    addConversion([](Type type) { return type; });
    addConversion([&](QASM::QubitType type) {
      return quantum::QubitType::get(this->context, 1);
    });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};

/// Base Pattern
template <typename SourceOp>
class QASMOpToQuantumConversionPattern : public OpConversionPattern<SourceOp> {
protected:
  QubitMap *qubitMap;
  MLIRContext *ctx;

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

/// qasm.pi : f*
/// [[to]]
/// constant [M_PI] : f*
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

/// %q = qasm.allocate
/// [[to]]
/// %q = qssa.allocate() : !qssa.qubit<1>
class AllocateOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::AllocateOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::AllocateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto qubitType = typeConverter->convertType(op.getType());
    auto allocOp = rewriter.create<quantum::AllocateOp>(
        rewriter.getUnknownLoc(), qubitType, ValueRange{});
    rewriter.replaceOp(op, allocOp.getResult());
    qubitMap->allocateQubit(op->getParentOfType<FuncOp>(), allocOp.getResult());
    return success();
  }
};

/// qasm.U(%theta : f*, %phi : f*, %lambda : f*) %q
/// [[to]]
/// %q_{i} =
///    qssa.U(%theta : f*, %phi : f*, %lambda : f*) %q_{i - 1} : !qssa.qubit<1>
class SingleQubitRotationOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::SingleQubitRotationOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::SingleQubitRotationOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::SingleQubitRotationOpAdaptor args(operands);
    auto func = op->getParentOfType<FuncOp>();
    auto qubit = qubitMap->resolveQubit(func, args.qinp());
    auto convertedOp = rewriter.create<quantum::UniversalRotationGateOp>(
        rewriter.getUnknownLoc(),
        typeConverter->convertType(op.qinp().getType()), args.theta(),
        args.phi(), args.lambda(), qubit);
    qubitMap->updateQubit(func, args.qinp(), convertedOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

/// qasm.CX %a, %b
/// [[to]]
/// %a_{i}, %b_{j} = qssa.CNOT %a_{i - 1} %b_{j - 1}
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

/// Convert a QASM gate declaration
/// Form: @<gate-name>([AnyFloat...], [QASM::QubitType...]) -> ()
/// [[to]]
/// @<gate-name>([AnyFloat...], [quantum::QubitType...])
///    -> ([quantum::QubitType...])
class FuncOpConversion : public QASMOpToQuantumConversionPattern<FuncOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // generate converted function type
    TypeConverter::SignatureConversion inputs(funcOp.getNumArguments());
    size_t qubitCount = 0;
    for (auto &en : llvm::enumerate(funcOp.getType().getInputs())) {
      inputs.addInputs(en.index(), typeConverter->convertType(en.value()));
      if (en.value().isa<QASM::QubitType>())
        qubitCount++;
    }
    TypeConverter::SignatureConversion results(qubitCount);
    qubitCount = 0;
    for (auto &en : llvm::enumerate(funcOp.getType().getInputs())) {
      if (en.value().isa<QASM::QubitType>())
        results.addInputs(qubitCount++, typeConverter->convertType(en.value()));
    }

    auto funcType =
        FunctionType::get(funcOp->getContext(), inputs.getConvertedTypes(),
                          results.getConvertedTypes());

    // Generate the new FuncOp, and inline the body region
    auto newFuncOp =
        rewriter.create<FuncOp>(funcOp->getLoc(), funcOp.getName(), funcType);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Convert the signature and delete the original operation
    rewriter.applySignatureConversion(&newFuncOp.getBody(), inputs);
    rewriter.eraseOp(funcOp);

    // populate the qubit map
    auto bodyArgs = newFuncOp.getBody().getArguments();
    for (auto arg : bodyArgs) {
      if (arg.getType().isa<quantum::QubitType>()) {
        qubitMap->allocateQubit(newFuncOp, arg, /*isFuncArgument=*/true);
      }
    }
    return success();
  }
};

/// Convert the return op for a qasm gate, returning the input qubits
class ReturnOpConversion : public QASMOpToQuantumConversionPattern<ReturnOp> {

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = returnOp->getParentOfType<FuncOp>();
    auto qubits = qubitMap->flattenResults(funcOp);
    rewriter.create<ReturnOp>(returnOp->getLoc(), qubits);
    rewriter.eraseOp(returnOp);
    return success();
  }
};

void populateQASMToQuantumConversionPatterns(
    QASMTypeConverter &typeConverter, QubitMap &qubitMap,
    OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      FuncOpConversion,
      ReturnOpConversion,
      PIOpConversion,
      AllocateOpConversion,
      SingleQubitRotationOpConversion,
      ControlledNotOpConversion
  >(typeConverter, &qubitMap);
  // clang-format on
}

struct QASMToQuantumTarget : public ConversionTarget {
  QASMToQuantumTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<quantum::QuantumDialect>();

    addIllegalDialect<QASM::QASMDialect>();
    addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp funcOp) -> bool { return !funcOp->hasAttr("qasm.gate"); });
    addDynamicallyLegalOp<CallOp>([&](CallOp callOp) -> bool {
      for (auto arg : callOp.getArgOperands()) {
        if (arg.getType().isa<QASM::QubitType>())
          return false;
      }
      return true;
    });
    addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp returnOp) { return !returnOp->hasAttr("qasm.gate_end"); });
  }
};

struct QASMToQuantumPass : public QASMToQuantumPassBase<QASMToQuantumPass> {
  void runOnOperation() override;
};

void QASMToQuantumPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
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
