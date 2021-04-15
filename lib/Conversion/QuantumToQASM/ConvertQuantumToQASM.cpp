#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversion/QuantumToQASM/Passes.h"
#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMOps.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "PassDetail.h"

using namespace mlir;

namespace {

// Maps qssa.qubit to the corresponding qasm.qubit
// For now only supports static size
class QubitMap {
public:
  QubitMap(MLIRContext *ctx) : ctx(ctx), qubits() {}
  void allocate(Value arg, ValueRange qfinal) {
    for (auto q : qfinal) {
      qubits[arg].push_back(q);
    }
  }
  void concat(Value lhs, Value rhs, Value res) {
    qubits[res] = qubits[lhs];
    qubits[res].insert(qubits[res].end(), qubits[rhs].begin(),
                       qubits[rhs].end());
  }
  void split(Value arg, Value resLhs, Value resRhs) {
    int lhsSize = resLhs.getType().cast<quantum::QubitType>().getSize();
    qubits[resLhs] = {qubits[arg].begin(), qubits[arg].begin() + lhsSize};
    qubits[resRhs] = {qubits[arg].begin() + lhsSize, qubits[arg].end()};
  };
  std::vector<Value> resolve(Value arg) { return qubits[arg]; }

  MLIRContext *getContext() const { return ctx; }

private:
  [[maybe_unused]] MLIRContext *ctx;
  llvm::DenseMap<Value, std::vector<Value>> qubits;
};

class QuantumTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QuantumTypeConverter(MLIRContext *context) : context(context) {
    addConversion([](Type type) { return type; });
    addConversion([&](quantum::QubitType type) {
      return QASM::QubitType::get(getContext());
    });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};

template <typename Op>
struct QuantumToQASMOpConversion : OpConversionPattern<Op> {
  QubitMap *qubitMap;
  Type getQASMQubitType() const {
    return QASM::QubitType::get(qubitMap->getContext());
  }

  QuantumToQASMOpConversion(MLIRContext *ctx, QubitMap *qubitMap)
      : OpConversionPattern<Op>(ctx), qubitMap(qubitMap) {}
};

struct AllocateOpConversion : QuantumToQASMOpConversion<quantum::AllocateOp> {
  using QuantumToQASMOpConversion::QuantumToQASMOpConversion;
  LogicalResult
  matchAndRewrite(quantum::AllocateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    int64_t size = op.getType().cast<quantum::QubitType>().getSize();
    std::vector<Value> qubits;
    for (int64_t i = 0; i < size; i++) {
      auto qubit =
          rewriter.create<QASM::AllocateOp>(op->getLoc(), getQASMQubitType());
      qubits.push_back(qubit);
    }
    qubitMap->allocate(op.getResult(), qubits);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateQuantumToQASMConversionPatterns(
    QuantumTypeConverter &typeConverter, QubitMap &qubitMap,
    OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      AllocateOpConversion
  >(patterns.getContext(), &qubitMap);
  // clang-format on
}

struct QuantumToQASMTarget : public ConversionTarget {
  QuantumToQASMTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<QASM::QASMDialect>();
    addLegalDialect<AffineDialect>();

    addIllegalDialect<quantum::QuantumDialect>();
  }
};

struct QuantumToQASMPass : public QuantumToQASMPassBase<QuantumToQASMPass> {
  void runOnOperation() override;
};

void QuantumToQASMPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  QuantumTypeConverter typeConverter(&getContext());
  QubitMap qubitMap(&getContext());
  populateQuantumToQASMConversionPatterns(typeConverter, qubitMap, patterns);

  QuantumToQASMTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createQuantumToQASMPass() {
  return std::make_unique<QuantumToQASMPass>();
}
} // namespace mlir
