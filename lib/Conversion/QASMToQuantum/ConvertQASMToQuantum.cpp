#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
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

/// Qubit Mapping: Stores the current qubit state in the circuit.
/// Access using the Value of the converted allocated qubit by `qssa.alloc`
class QubitMap {
  [[maybe_unused]] MLIRContext *ctx;
  llvm::StringMap<DenseMap<Value, Value>> mapping;
  llvm::StringMap<DenseMap<Value, Value>> qasmMapping;
  llvm::StringMap<SmallVector<Value>> functionArguments;

public:
  QubitMap(MLIRContext *ctx)
      : ctx(ctx), mapping(), qasmMapping(), functionArguments() {}

  /// add a new allocated qubit
  /// if it is a function argument, mark it, for flattening the return values
  void allocateQubit(FuncOp func, Value qubit, bool isFuncArgument = false) {
    mapping[func.getName()][qubit] = qubit;
    if (isFuncArgument)
      functionArguments[func.getName()].push_back(qubit);
  }
  /// Set mapping from qasm.alloc to qssa.alloc qubit results
  void setMapping(FuncOp func, Value qasmQubit, Value qssaQubit) {
    qasmMapping[func.getName()][qasmQubit] = qssaQubit;
  }

  /// get the current state of a qubit
  Value resolveQubit(FuncOp func, Value qubit) {
    return mapping[func.getName()][qubit];
  }
  /// get the qssa base qubit for the given qasm qubit
  Value resolveQASMQubit(FuncOp func, Value qasmQubit) {
    return qasmMapping[func.getName()][qasmQubit];
  }

  // Find the base qubit (alloc) given an intermediate SSA qubit
  Value inverseLookup(FuncOp func, Value finalQubit) {
    for (auto &elem : mapping[func.getName()]) {
      if (elem.getSecond() == finalQubit)
        return elem.getFirst();
    }
    assert(false && "invalid qubit inverse lookup");
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
    qubitMap->setMapping(op->getParentOfType<FuncOp>(), op.qout(),
                         allocOp.getResult());
    return success();
  }
};

/// %res = qasm.measure %q
/// [[to]]
/// %res, %q_{i + 1} = qssa.measure_one %q_{i}
class MeasureOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::MeasureOp> {
public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::MeasureOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::MeasureOpAdaptor args(operands);
    auto parentFuncOp = op->getParentOfType<FuncOp>();
    auto currentQubit = qubitMap->resolveQubit(parentFuncOp, args.qinp());
    auto newOp =
        rewriter.create<quantum::MeasureQubitOp>(op->getLoc(), currentQubit);
    rewriter.replaceOp(op, newOp.getResult(0));
    qubitMap->updateQubit(parentFuncOp, args.qinp(), newOp.getResult(1));
    return success();
  }
};

/// qasm.reset %q
/// [[to]]
/// %ign = qssa.measure_one %q_{i}
/// %q_{i + 1} = qssa.allocate() : !qssa.qubit<1>
class ResetOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::ResetOp> {
public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::ResetOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::ResetOpAdaptor args(operands);
    auto parentFuncOp = op->getParentOfType<FuncOp>();
    auto currentQubit = qubitMap->resolveQubit(parentFuncOp, args.qinp());
    rewriter.create<quantum::MeasureQubitOp>(op->getLoc(), currentQubit);
    auto newQubit =
        rewriter.create<quantum::AllocateOp>(rewriter.getUnknownLoc(), 1);
    qubitMap->updateQubit(parentFuncOp, args.qinp(), newQubit.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

/// qasm.barrier %q
/// [[to]]
/// %q_{i + 1} = qssa.barrier %q_{i} : !qssa.qubit<1>
class BarrierOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::BarrierOp> {
public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::BarrierOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::BarrierOpAdaptor args(operands);
    auto parentFuncOp = op->getParentOfType<FuncOp>();
    auto currentQubit = qubitMap->resolveQubit(parentFuncOp, args.qinp());
    auto newQubit = rewriter.create<quantum::BarrierOp>(
        rewriter.getUnknownLoc(), currentQubit.getType(), currentQubit);
    qubitMap->updateQubit(parentFuncOp, args.qinp(), newQubit.getResult());
    rewriter.eraseOp(op);
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
    newFuncOp.setPrivate();
    if (funcOp->hasAttrOfType<StringAttr>("qasm.stdgate"))
      newFuncOp->setAttr("qasm.stdgate",
                         funcOp->getAttrOfType<StringAttr>("qasm.stdgate"));

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

class CallOpConversion : public QASMOpToQuantumConversionPattern<CallOp> {
public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(CallOp callOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CallOpAdaptor resolvedOperands(operands);
    auto parentFuncOp = callOp->getParentOfType<FuncOp>();

    // qubits to return
    SmallVector<Type> resultTypes;
    SmallVector<Value> arguments, baseQubits;
    for (auto arg : resolvedOperands.getOperands()) {
      if (arg.getType().isa<quantum::QubitType>()) {
        resultTypes.push_back(arg.getType());
        auto qubit = qubitMap->resolveQubit(parentFuncOp, arg);
        arguments.push_back(qubit);
        baseQubits.push_back(arg);
      } else {
        arguments.push_back(arg);
      }
    }

    // generate new call
    auto newCallOp = rewriter.create<CallOp>(
        callOp->getLoc(), callOp.getCallee(), resultTypes, arguments);
    rewriter.eraseOp(callOp);

    // update qubit map
    for (auto qubitPair : llvm::zip(baseQubits, newCallOp.getResults())) {
      Value baseQubit, finalQubit;
      std::tie(baseQubit, finalQubit) = qubitPair;
      qubitMap->updateQubit(parentFuncOp, baseQubit, finalQubit);
    }

    return success();
  }
};

class GateCallOpConversion
    : public QASMOpToQuantumConversionPattern<QASM::GateCall> {
  template <class GateOp>
  Value insertSimplePrimitiveGateOp(Location loc, Value inputQubit,
                                    ConversionPatternRewriter &rewriter) const {
    auto gateOp =
        rewriter.create<GateOp>(loc, inputQubit.getType(), inputQubit);
    return gateOp.qout();
  }

public:
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::GateCall gateOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    QASM::GateCallAdaptor resolvedOperands(operands);
    auto parentFuncOp = gateOp->getParentOfType<FuncOp>();

    // qubits to return
    SmallVector<Type> resultTypes;
    SmallVector<Value> arguments, baseQubits;
    for (auto arg : resolvedOperands.getOperands()) {
      if (arg.getType().isa<quantum::QubitType>()) {
        resultTypes.push_back(arg.getType());
        auto qubit = qubitMap->resolveQubit(parentFuncOp, arg);
        arguments.push_back(qubit);
        baseQubits.push_back(arg);
      } else {
        arguments.push_back(arg);
      }
    }

    SmallVector<Value> resultQubits;

    // currently supported: x, y, z, s, sdg, t, tdg, rx, ry, rz
    if (gateOp.gate_name() == "x") {
      resultQubits.push_back(insertSimplePrimitiveGateOp<quantum::PauliXGateOp>(
          gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "y") {
      resultQubits.push_back(insertSimplePrimitiveGateOp<quantum::PauliYGateOp>(
          gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "z") {
      resultQubits.push_back(insertSimplePrimitiveGateOp<quantum::PauliZGateOp>(
          gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "h") {
      resultQubits.push_back(
          insertSimplePrimitiveGateOp<quantum::HadamardGateOp>(
              gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "s") {
      resultQubits.push_back(insertSimplePrimitiveGateOp<quantum::PhaseGateOp>(
          gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "sdg") {
      resultQubits.push_back(
          insertSimplePrimitiveGateOp<quantum::PhaseDaggerGateOp>(
              gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "t") {
      resultQubits.push_back(insertSimplePrimitiveGateOp<quantum::TGateOp>(
          gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "tdg") {
      resultQubits.push_back(
          insertSimplePrimitiveGateOp<quantum::TDaggerGateOp>(
              gateOp->getLoc(), arguments[0], rewriter));
    } else if (gateOp.gate_name() == "rx") {
      resultQubits.push_back(rewriter.create<quantum::RotateXOp>(
          gateOp->getLoc(), arguments[1].getType(), arguments[0],
          arguments[1]));
    } else if (gateOp.gate_name() == "ry") {
      resultQubits.push_back(rewriter.create<quantum::RotateYOp>(
          gateOp->getLoc(), arguments[1].getType(), arguments[0],
          arguments[1]));
    } else if (gateOp.gate_name() == "rz") {
      resultQubits.push_back(rewriter.create<quantum::RotateZOp>(
          gateOp->getLoc(), arguments[1].getType(), arguments[0],
          arguments[1]));
    } else {
      // generate new call
      emitWarning(gateOp->getLoc())
          << "Unknown gate call, converting to std.call instead";
      auto newCallOp = rewriter.create<CallOp>(
          gateOp->getLoc(), gateOp.gate_name(), resultTypes, arguments);
      resultQubits = newCallOp.getResults();
    }

    rewriter.eraseOp(gateOp);

    // update qubit map
    for (auto qubitPair : llvm::zip(baseQubits, resultQubits)) {
      Value baseQubit, finalQubit;
      std::tie(baseQubit, finalQubit) = qubitPair;
      qubitMap->updateQubit(parentFuncOp, baseQubit, finalQubit);
    }

    return success();
  }
};

struct SCFIfConversion : QASMOpToQuantumConversionPattern<scf::IfOp> {
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentFuncOp = ifOp->getParentOfType<FuncOp>();

    llvm::SmallVector<Value> qasmQubits, qssaQubits;
    llvm::SmallVector<Type> yieldTypes;
    // ASSUMPTION: EXACTLY one quantum operation in the thenRegion;
    // elseRegion is empty.
    for (auto &inst : ifOp.thenRegion().getBlocks().begin()->getOperations()) {
      if (inst.hasTrait<OpTrait::IsTerminator>())
        break;
      for (auto arg : inst.getOperands()) {
        if (arg.getType().isa<QASM::QubitType>()) {
          qasmQubits.push_back(arg);
          auto baseQubit = qubitMap->resolveQASMQubit(parentFuncOp, arg);
          qssaQubits.push_back(baseQubit);
          yieldTypes.push_back(getTypeConverter()->convertType(arg.getType()));
        }
      }
    }

    auto newIfOp =
        rewriter.create<scf::IfOp>(ifOp->getLoc(), yieldTypes, ifOp.condition(),
                                   /*withElseRegion=*/true);
    auto pendingThenBuilder =
        newIfOp.getThenBodyBuilder(rewriter.getListener());
    auto elseBuilder = newIfOp.getElseBodyBuilder(); // do not listen/update
    for (auto &inst : ifOp.thenRegion().getBlocks().begin()->getOperations()) {
      if (inst.hasTrait<OpTrait::IsTerminator>()) {
        pendingThenBuilder.create<scf::YieldOp>(inst.getLoc(), qssaQubits)
            ->setAttr("qasm.if", rewriter.getUnitAttr());
        SmallVector<Value> resolvedQubits(
            llvm::map_range(qssaQubits, [&](Value v) {
              return qubitMap->resolveQubit(parentFuncOp, v);
            }));
        elseBuilder.create<scf::YieldOp>(inst.getLoc(), resolvedQubits);
      } else {
        pendingThenBuilder.insert(inst.clone());
      }
    }

    rewriter.eraseOp(ifOp);
    return success();
  }
};

struct SCFYieldConversion : QASMOpToQuantumConversionPattern<scf::YieldOp> {
  using QASMOpToQuantumConversionPattern::QASMOpToQuantumConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp yieldOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto parentFuncOp = yieldOp->getParentOfType<FuncOp>();
    auto ifOp = yieldOp->getParentOfType<scf::IfOp>();

    SmallVector<Value> finalQubits;
    for (auto argPair : llvm::zip(yieldOp.results(), ifOp.results())) {
      Value baseQubit, resultQubit;
      std::tie(baseQubit, resultQubit) = argPair;
      finalQubits.push_back(qubitMap->resolveQubit(parentFuncOp, baseQubit));
      qubitMap->updateQubit(parentFuncOp, baseQubit, resultQubit);
    }
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), finalQubits);
    rewriter.eraseOp(yieldOp);
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
      CallOpConversion,
      GateCallOpConversion,
      AllocateOpConversion,
      MeasureOpConversion,
      ResetOpConversion,
      BarrierOpConversion,
      SingleQubitRotationOpConversion,
      ControlledNotOpConversion,
      SCFIfConversion,
      SCFYieldConversion
  >(typeConverter, &qubitMap);
  // clang-format on
}

struct QASMToQuantumTarget : public ConversionTarget {
  QASMToQuantumTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<quantum::QuantumDialect>();
    addLegalDialect<AffineDialect>();
    addLegalDialect<scf::SCFDialect>();

    addIllegalDialect<QASM::QASMDialect>();
    addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp funcOp) -> bool { return !funcOp->hasAttr("qasm.gate"); });
    addDynamicallyLegalOp<CallOp>(
        [&](CallOp callOp) -> bool { return !callOp->hasAttr("qasm.gate"); });
    addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp returnOp) { return !returnOp->hasAttr("qasm.gate_end"); });
    addDynamicallyLegalOp<scf::IfOp>(
        [&](scf::IfOp op) { return !op->hasAttr("qasm.if"); });
    addDynamicallyLegalOp<scf::YieldOp>(
        [&](scf::YieldOp op) { return !op->hasAttr("qasm.if"); });
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
