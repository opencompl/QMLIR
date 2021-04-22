//===- ConvertQuantumToStandard.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversion/QuantumToLLVM/ConvertQuantumToLLVM.h"
#include "Conversion/QuantumToLLVM/Passes.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/Quantum/QuantumTypes.h"
#include "PassDetail.h"

using namespace mlir;
using namespace quantum;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class QuantumToLLVMTarget : public ConversionTarget {
public:
  explicit QuantumToLLVMTarget(MLIRContext &context)
      : ConversionTarget(context) {
    addLegalDialect<AffineDialect>();
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<scf::SCFDialect>();

    addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
      auto funcType = funcOp.getType();
      for (auto &arg : llvm::enumerate(funcType.getInputs())) {
        if (arg.value().isa<QubitType>())
          return false;
      }
      for (auto &arg : llvm::enumerate(funcType.getResults())) {
        if (arg.value().isa<QubitType>())
          return false;
      }
      return true;
    });

    addIllegalDialect<QuantumDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct QuantumToLLVMPass : public QuantumToLLVMPassBase<QuantumToLLVMPass> {
  void runOnOperation() override;
};

void QuantumToLLVMPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  QuantumTypeConverter typeConverter(&getContext());
  populateQuantumToLLVMConversionPatterns(typeConverter, patterns);
  QuantumToLLVMTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace quantum {

// Populate the conversion pattern list
void populateQuantumToLLVMConversionPatterns(
    QuantumTypeConverter &typeConverter,
    mlir::OwningRewritePatternList &patterns) {
  // TODO: Add QIR Conversion Patterns
  // clang-format off
  // patterns.insert<
  // >(typeConverter);
  // clang-format on
}

QuantumTypeConverter::QuantumTypeConverter(MLIRContext *context)
    : context(context) {
  // Add type conversions
  addConversion([](Type type) { return type; });
  // TODO: Add QIR Type conversions
}

} // namespace quantum
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertQuantumToLLVMPass() {
  return std::make_unique<QuantumToLLVMPass>();
}
