#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversion/ZXToQuantum/Passes.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "PassDetail.h"

using namespace mlir;
using quantum::QubitType;
using ZX::WireType;

namespace {

class ZXTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  ZXTypeConverter(MLIRContext *context) : context(context) {
    addConversion([](Type type) { return type; });
    addConversion(
        [&](WireType type) { return QubitType::get(type.getContext(), 1); });
  }
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};
/// TODO: Implement support for multiqubit arrays

//====== PATTERNS ======

void populateZXToQuantumConversionPatterns(ZXTypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  // clang-format off
  // patterns.insert<
  // >(typeConverter, typeConverter.getContext());
  // clang-format on
}

struct ZXToQuantumTarget : public ConversionTarget {
  ZXToQuantumTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<quantum::QuantumDialect>();
    addIllegalDialect<ZX::ZXDialect>();
  }
};

struct ZXToQuantumPass : public ZXToQuantumPassBase<ZXToQuantumPass> {
  void runOnOperation() override;
};

void ZXToQuantumPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  ZXTypeConverter typeConverter(&getContext());
  populateZXToQuantumConversionPatterns(typeConverter, patterns);

  ZXToQuantumTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createZXToQuantumPass() {
  return std::make_unique<ZXToQuantumPass>();
}
} // namespace mlir
