#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMOps.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

class QASMTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  QASMTypeConverter(MLIRContext *context) : context(context) {
    addConversion([](Type type) { return type; });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};

void populateQASMToSCFConversionPatterns(QASMTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  // clang-format off
  //patterns.insert<
  //>(typeConverter);
  // clang-format on
}

struct QASMToSCFTarget : public ConversionTarget {
  QASMToSCFTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<scf::SCFDialect>();

    addIllegalOp<QASM::IfOp>();
  }
};

struct QASMToSCFPass : public QASMToSCFPassBase<QASMToSCFPass> {
  void runOnOperation() override;
};

void QASMToSCFPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  QASMTypeConverter typeConverter(&getContext());
  populateQASMToSCFConversionPatterns(typeConverter, patterns);

  QASMToSCFTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createQASMToSCFPass() {
  return std::make_unique<QASMToSCFPass>();
}
} // namespace mlir
