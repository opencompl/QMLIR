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

struct QASMIfOpConversion : OpConversionPattern<QASM::IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(QASM::IfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    int64_t regSize = op.creg().getType().cast<MemRefType>().getShape()[0];
    auto trueOp =
        rewriter.create<ConstantIntOp>(rewriter.getUnknownLoc(), 1, 1);
    Value cond = trueOp;
    for (int64_t i = 0; i < regSize; i++) {
      Value bit = rewriter.create<AffineLoadOp>(
          op->getLoc(), op.creg(), rewriter.getConstantAffineMap(i),
          ValueRange{});
      if ((op.value() >> i) & 1) {
        bit = rewriter.create<XOrOp>(op->getLoc(), bit.getType(), bit, trueOp);
      }
      cond = rewriter.create<AndOp>(op->getLoc(), bit.getType(), cond, bit);
    }
    auto scfIfOp = rewriter.create<scf::IfOp>(op->getLoc(), cond, false);
    scfIfOp->setAttr("qasm.if", rewriter.getUnitAttr());
    auto thenBuilder = scfIfOp.getThenBodyBuilder();
    for (auto &inst : op.ifBlock().getBlocks().begin()->getOperations()) {
      if (inst.hasTrait<OpTrait::IsTerminator>())
        break;
      thenBuilder.insert(inst.clone());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateQASMToSCFConversionPatterns(QASMTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
    QASMIfOpConversion
  >(typeConverter, patterns.getContext());
  // clang-format on
}

struct QASMToSCFTarget : public ConversionTarget {
  QASMToSCFTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<AffineDialect>();

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
