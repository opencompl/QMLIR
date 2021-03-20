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

//====== PATTERNS ======
class PIOpLowering : public OpConversionPattern<QASM::PIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
};

void populateQASMToQuantumConversionPatterns(
    QASMTypeConverter &typeConverter, OwningRewritePatternList &patterns) {
  patterns.insert<PIOpLowering>(typeConverter, typeConverter.getContext());
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
