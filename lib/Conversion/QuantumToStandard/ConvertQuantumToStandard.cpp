//===- ConvertQuantumToStandard.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/QuantumToStandard/ConvertQuantumToStandard.h"
#include "Conversion/QuantumToStandard/Passes.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/Quantum/QuantumTypes.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace quantum;

namespace {

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Func Op Lowering
//===----------------------------------------------------------------------===//

class FuncOpLowering : public QuantumOpToStdPattern<FuncOp> {
public:
  using QuantumOpToStdPattern<FuncOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto funcOp = cast<FuncOp>(operation);

    // Convert the original function arguments
    TypeConverter::SignatureConversion inputs(funcOp.getNumArguments());
    for (auto &en : llvm::enumerate(funcOp.getType().getInputs()))
      inputs.addInputs(en.index(), typeConverter.convertType(en.value()));

    TypeConverter::SignatureConversion results(funcOp.getNumResults());
    for (auto &en : llvm::enumerate(funcOp.getType().getResults()))
      results.addInputs(en.index(), typeConverter.convertType(en.value()));

    auto funcType =
        FunctionType::get(inputs.getConvertedTypes(),
                          results.getConvertedTypes(), funcOp.getContext());

    // Replace the function by a function with an updated signature
    auto newFuncOp =
        rewriter.create<FuncOp>(loc, funcOp.getName(), funcType, llvm::None);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Convert the signature and delete the original operation
    rewriter.applySignatureConversion(&newFuncOp.getBody(), inputs);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Allocate Op Lowering
//===----------------------------------------------------------------------===//

class AllocateOpLowering : public QuantumOpToStdPattern<AllocateOp> {
public:
  using QuantumOpToStdPattern<AllocateOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the acquire_qubits function, or declare if it doesn't exist.
    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    auto acquireFuncType = rewriter.getFunctionType(
        {rewriter.getIndexType()},
        {MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI64Type())});
    auto acquireFunc =
        module.lookupSymbol<FuncOp>("__mlir_quantum_simulator__acquire_qubits");
    if (!acquireFunc) {
      // Declare the acquire_qubits function
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      acquireFunc = rewriter.create<FuncOp>(
          rewriter.getUnknownLoc(), "__mlir_quantum_simulator__acquire_qubits",
          acquireFuncType);
    }

    auto allocateOp = cast<AllocateOp>(operation);
    auto qubitType = allocateOp.getResult().getType().cast<QubitType>();

    if (qubitType.hasStaticSize()) {
      rewriter.setInsertionPoint(operation);
      auto sizeOp = rewriter.create<ConstantIndexOp>(rewriter.getUnknownLoc(),
                                                     qubitType.getSize());
      auto qubitSize = sizeOp.getResult();
      auto acquireCallOp = rewriter.create<CallOp>(
          rewriter.getUnknownLoc(), acquireFunc, ValueRange{qubitSize});
      auto castOp = rewriter.create<MemRefCastOp>(
          rewriter.getUnknownLoc(), acquireCallOp.getResults()[0],
          typeConverter.convertType(qubitType));
      rewriter.replaceOp(operation, castOp.getResult());
    } else {
      // pass size to `acquire_qubits`
      auto qubitSize = allocateOp.getOperand(0);
      auto acquireCallOp = rewriter.create<CallOp>(
          rewriter.getUnknownLoc(), acquireFunc, ValueRange{qubitSize});
      rewriter.replaceOp(operation, acquireCallOp.getResults());
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Dimension Op Lowering
//===----------------------------------------------------------------------===//

class DimensionOpLowering : public QuantumOpToStdPattern<DimensionOp> {
public:
  using QuantumOpToStdPattern<DimensionOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    //    auto dimensionOp = cast<DimensionOp>(operation);
    DimensionOp::Adaptor transformed(operands);

    auto convertedDimOp = rewriter.create<::mlir::DimOp>(
        rewriter.getUnknownLoc(), transformed.getODSOperands(0).front(), 0);

    rewriter.replaceOp(operation,
                       ValueRange{transformed.getODSOperands(0).front(),
                                  convertedDimOp.getResult()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cast Op Lowering
//===----------------------------------------------------------------------===//

class CastOpLowering : public QuantumOpToStdPattern<CastOp> {
public:
  using QuantumOpToStdPattern<CastOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto castOp = cast<CastOp>(operation);
    CastOp::Adaptor transformed(operands);

    auto dstType = castOp.getType();

    auto convertedCastOp = rewriter.create<MemRefCastOp>(
        rewriter.getUnknownLoc(), transformed.getODSOperands(0)[0],
        typeConverter.convertType(dstType));

    rewriter.replaceOp(castOp, convertedCastOp.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Concat Op Lowering
//===----------------------------------------------------------------------===//

class ConcatOpLowering : public QuantumOpToStdPattern<ConcatOp> {
public:
  using QuantumOpToStdPattern<ConcatOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the concat_qubits function, or declare if it doesn't exist.
    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    auto qLibMemRefType =
        MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI64Type());
    auto concatFuncType = rewriter.getFunctionType(
        {qLibMemRefType, qLibMemRefType}, {qLibMemRefType});
    auto concatFunc =
        module.lookupSymbol<FuncOp>("__mlir_quantum_simulator__concat_qubits");
    if (!concatFunc) {
      // Declare the concat_qubits function
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      concatFunc = rewriter.create<FuncOp>(
          rewriter.getUnknownLoc(), "__mlir_quantum_simulator__concat_qubits",
          concatFuncType);
    }

    auto concatOp = cast<ConcatOp>(operation);
    ConcatOp::Adaptor transformed(operands);

    // Convert operands to dynamic size (for library call compatibility)
    SmallVector<Value, 2> convertedOperands;
    for (auto en : transformed.getODSOperands(0)) {
      auto argType = en.getType().cast<MemRefType>();
      if (argType.hasStaticShape()) {
        auto castOp = rewriter.create<MemRefCastOp>(rewriter.getUnknownLoc(),
                                                    en, qLibMemRefType);
        convertedOperands.push_back(castOp.getResult());
      } else {
        convertedOperands.push_back(en);
      }
    }

    // call library concat function
    auto concatLibCall = rewriter.create<CallOp>(
        rewriter.getUnknownLoc(), concatFunc, ValueRange(convertedOperands));

    auto resultQubitType = concatOp.getType().cast<QubitType>();
    if (resultQubitType.hasStaticSize()) {
      auto resultCastOp = rewriter.create<MemRefCastOp>(
          rewriter.getUnknownLoc(), concatLibCall.getResult(0),
          typeConverter.convertType(resultQubitType));
      rewriter.replaceOp(operation, resultCastOp.getResult());
    } else {
      rewriter.replaceOp(operation, concatLibCall.getResult(0));
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Split Op Lowering
//===----------------------------------------------------------------------===//

class SplitOpLowering : public QuantumOpToStdPattern<SplitOp> {
public:
  using QuantumOpToStdPattern<SplitOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the split function, or declare if it doesn't exist.
    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    auto qLibMemRefType =
        MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI64Type());
    auto qLibIndexType = rewriter.getIndexType();
    auto splitFuncType =
        rewriter.getFunctionType({qLibMemRefType, qLibIndexType, qLibIndexType},
                                 {qLibMemRefType, qLibMemRefType});
    auto splitFunc =
        module.lookupSymbol<FuncOp>("__mlir_quantum_simulator__split_qubits");
    if (!splitFunc) {
      // Declare the split function
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      splitFunc = rewriter.create<FuncOp>(
          rewriter.getUnknownLoc(), "__mlir_quantum_simulator__split_qubits",
          splitFuncType);
    }

    auto splitOp = cast<SplitOp>(operation);
    SplitOp::Adaptor transformed(operands);

    rewriter.setInsertionPoint(operation);

    SmallVector<Value, 3> splitLibCallOperands;
    // Convert operand to dynamic size (for library call compatibility)
    Value convertedOperand = transformed.getODSOperands(0).front();
    if (convertedOperand.getType().cast<MemRefType>().hasStaticShape()) {
      auto castOp = rewriter.create<MemRefCastOp>(
          rewriter.getUnknownLoc(), convertedOperand, qLibMemRefType);
      convertedOperand = castOp.getResult();
    }
    splitLibCallOperands.push_back(convertedOperand);

    // Get split-size index operands
    auto indexOperandRange = transformed.getODSOperands(1);
    auto indexValueIter = indexOperandRange.begin();
    for (auto en : llvm::enumerate(splitOp.getResults())) {
      auto qubitType = en.value().getType().cast<QubitType>();
      if (qubitType.hasStaticSize()) {
        // create a constant, to pass to splitLibCall
        auto indexOp = rewriter.create<ConstantIndexOp>(
            rewriter.getUnknownLoc(), qubitType.getSize());
        splitLibCallOperands.push_back(indexOp.getResult());
      } else {
        // use a provided size operand
        assert(indexValueIter != indexOperandRange.end() &&
               "not enough index operands");
        splitLibCallOperands.push_back(*indexValueIter);
        indexValueIter++;
      }
    }
    assert(indexValueIter == indexOperandRange.end() &&
           "unused index operands");

    // call library split function
    auto splitLibCall = rewriter.create<CallOp>(
        rewriter.getUnknownLoc(), splitFunc, ValueRange(splitLibCallOperands));

    // Un-cast static sized arrays
    SmallVector<Value, 2> results;
    for (auto en : llvm::enumerate(splitOp.getResults())) {
      if (en.value().getType().cast<QubitType>().hasStaticSize()) {
        auto resultCastOp = rewriter.create<MemRefCastOp>(
            rewriter.getUnknownLoc(), splitLibCall.getResult(en.index()),
            typeConverter.convertType(en.value().getType()));
        results.push_back(resultCastOp.getResult());
      } else {
        results.push_back(splitLibCall.getResult(en.index()));
      }
    }

    rewriter.replaceOp(operation, results);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Measure Op Lowering
//===----------------------------------------------------------------------===//

class MeasureOpLowering : public QuantumOpToStdPattern<MeasureOp> {
public:
  using QuantumOpToStdPattern<MeasureOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the measure function, or declare if it doesn't exist.
    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    auto qLibMemRefType =
        MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI64Type());
    auto measureFuncType = rewriter.getFunctionType(
        {qLibMemRefType},
        {MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI1Type())});
    auto measureFunc =
        module.lookupSymbol<FuncOp>("__mlir_quantum_simulator__measure_qubits");
    if (!measureFunc) {
      // Declare the measure function
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      measureFunc = rewriter.create<FuncOp>(
          rewriter.getUnknownLoc(), "__mlir_quantum_simulator__measure_qubits",
          measureFuncType);
    }

    auto measureOp = cast<MeasureOp>(operation);
    MeasureOp::Adaptor transformed(operands);

    // Convert operand to dynamic size (for library call compatibility)
    SmallVector<Value, 1> convertedOperands;
    for (unsigned i = 0; i < 1; i++) {
      auto currentOperand = transformed.getODSOperands(i).front();
      auto argType = currentOperand.getType().cast<MemRefType>();
      if (argType.hasStaticShape()) {
        auto castOp = rewriter.create<MemRefCastOp>(
            rewriter.getUnknownLoc(), currentOperand, qLibMemRefType);
        convertedOperands.push_back(castOp.getResult());
      } else {
        convertedOperands.push_back(currentOperand);
      }
    }

    // call library measure function
    auto measureLibCall = rewriter.create<CallOp>(
        rewriter.getUnknownLoc(), measureFunc, ValueRange(convertedOperands));

    auto resultType = measureOp.getType().cast<MemRefType>();
    if (resultType.hasStaticShape()) {
      auto resultCastOp = rewriter.create<MemRefCastOp>(
          rewriter.getUnknownLoc(), measureLibCall.getResult(0), resultType);
      rewriter.replaceOp(operation, resultCastOp.getResult());
    } else {
      rewriter.replaceOp(operation, measureLibCall.getResult(0));
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Primitive Gate Op Lowering
//===----------------------------------------------------------------------===//

template <typename PrimitiveGateOp>
class PrimitiveGateOpLowering : public QuantumOpToStdPattern<PrimitiveGateOp> {
  static_assert(
      llvm::is_one_of<PrimitiveGateOp, PauliXGateOp, PauliYGateOp, PauliZGateOp,
                      HadamardGateOp, CNOTGateOp>::value,
      "invalid gate OP");

public:
  using QuantumOpToStdPattern<PrimitiveGateOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto primitiveGateOp = cast<PrimitiveGateOp>(operation);
    typename PrimitiveGateOp::Adaptor transformed(operands);

    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    // Find the corresponding gate function, or declare if it doesn't exist.
    auto qLibMemRefType =
        MemRefType::get({MemRefType::kDynamicSize}, rewriter.getI64Type());
    auto gateFuncType =
        rewriter.getFunctionType({qLibMemRefType}, {qLibMemRefType});

    // get the operation name, without the leading `quantum.`
    StringRef opName = primitiveGateOp.getOperationName().split('.').second;
    std::string name = std::string("__mlir_quantum_simulator__gate_") +
                       std::string(opName.data());
    auto gateFunc = module.lookupSymbol<FuncOp>(StringRef(name));

    if (!gateFunc) {
      // Declare the gate function
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      gateFunc =
          rewriter.create<FuncOp>(rewriter.getUnknownLoc(), name, gateFuncType);
    }

    // Convert operand to dynamic size (for library call compatibility)
    SmallVector<Value, 1> convertedOperands;
    auto inputQubit = transformed.getODSOperands(1).front();
    auto argType = inputQubit.getType().template cast<MemRefType>();
    if (argType.hasStaticShape()) {
      auto castOp = rewriter.create<MemRefCastOp>(rewriter.getUnknownLoc(),
                                                  inputQubit, qLibMemRefType);
      convertedOperands.push_back(castOp.getResult());
    } else {
      convertedOperands.push_back(inputQubit);
    }

    // call library gate function
    auto gateLibCall = rewriter.create<CallOp>(
        rewriter.getUnknownLoc(), gateFunc, ValueRange(convertedOperands));

    auto resultType = primitiveGateOp.getType().template cast<QubitType>();
    if (resultType.hasStaticSize()) {
      auto resultCastOp = rewriter.create<MemRefCastOp>(
          rewriter.getUnknownLoc(), gateLibCall.getResult(0),
          this->typeConverter.convertType(resultType));
      rewriter.replaceOp(operation, resultCastOp.getResult());
    } else {
      rewriter.replaceOp(operation, gateLibCall.getResult(0));
    }

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
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
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
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalDialect<QuantumDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
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
  patterns.insert<FuncOpLowering,

                  // Quantum Ops
                  AllocateOpLowering, CastOpLowering, DimensionOpLowering,
                  ConcatOpLowering, SplitOpLowering, MeasureOpLowering,

                  // Quantum Primitive Gate Ops
                  PrimitiveGateOpLowering<PauliXGateOp>,
                  PrimitiveGateOpLowering<PauliYGateOp>,
                  PrimitiveGateOpLowering<PauliZGateOp>,
                  PrimitiveGateOpLowering<HadamardGateOp>,
                  PrimitiveGateOpLowering<CNOTGateOp>>(typeConverter);
}

//===----------------------------------------------------------------------===//
// Quantum Type Converter
//===----------------------------------------------------------------------===//

QuantumTypeConverter::QuantumTypeConverter(MLIRContext *context)
    : context(context) {
  // Add type conversions
  addConversion([&](QubitType qubitType) -> Type {
    return MemRefType::get(qubitType.getMemRefShape(),
                           qubitType.getMemRefType());
  });
  addConversion([&](Type type) -> Optional<Type> {
    if (type.isa<QubitType>())
      return llvm::None;
    return type;
  });
}

//===----------------------------------------------------------------------===//
// Quantum Pattern Base Class
//===----------------------------------------------------------------------===//

QuantumToStdPattern::QuantumToStdPattern(StringRef rootOpName,
                                         QuantumTypeConverter &typeConverter,
                                         PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
      typeConverter(typeConverter) {}

} // namespace quantum
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertQuantumToStandardPass() {
  return std::make_unique<QuantumToStandardPass>();
}
