#include "QuantumToStandard/ConvertQuantumToStandard.h"
#include "QuantumToStandard/Passes.h"
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"
#include "Quantum/QuantumTypes.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace quantum;
using namespace scf;

namespace {

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

class FuncOpLowering : public QuantumOpToStdPattern<FuncOp> {
public:
  using QuantumOpToStdPattern<FuncOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};


//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class QuantumToStdTarget : public ConversionTarget {
public:
  explicit QuantumToStdTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct QuantumToStandardPass
    : public QuantumToStandardPassBase<QuantumToStandardPass> {
  void runOnOperation() override;
};

void QuantumToStandardPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto module = getOperation();

  QuantumTypeConverter typeConverter(module.getContext());
  populateQuantumToStdConversionPatterns(typeConverter, patterns);

  QuantumToStdTarget target(*(module.getContext()));
  
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<SCFDialect>();
  target.addDynamicallyLegalOp<FuncOp>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalDialect<QuantumDialect>();
  
  if (failed(applyFullConversion(module, target, patterns))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace quantum {

// Populate the conversion pattern list
void populateQuantumToStdConversionPatterns(
    QuantumTypeConverter &typeConverter,
    mlir::OwningRewritePatternList &patterns) {
  // patterns.insert</* OPS HERE */>(typeConverter);
}

//===----------------------------------------------------------------------===//
// Quantum Type Converter
//===----------------------------------------------------------------------===//

QuantumTypeConverter::QuantumTypeConverter(MLIRContext *context_)
    : context(context_) {
  // Add type conversions
}

//===----------------------------------------------------------------------===//
// Quantum Pattern Base Class
//===----------------------------------------------------------------------===//

QuantumToStdPattern::QuantumToStdPattern(StringRef rootOpName,
                                         QuantumTypeConverter &typeConverter_,
                                         PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter_.getContext()),
      typeConverter(typeConverter_) {}

} // namespace quantum
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertQuantumToStandardPass() {
  return std::make_unique<QuantumToStandardPass>();
}
