//===- ConvertQuantumToStandard.h -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM_H
#define CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM_H

#include "Dialect/Quantum/QuantumOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quantum {

/// Convert quantum types to QIR/LLVM types
struct QuantumTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  /// Create a quantum type converter using the default conversions
  QuantumTypeConverter(MLIRContext *context);

  MLIRContext *getContext() { return context; }

private:
  MLIRContext *context;
};

template <typename OpTy>
struct QuantumOpToLLVMPattern : public OpConversionPattern<OpTy> {
  QuantumOpToLLVMPattern(QuantumTypeConverter &typeConverter,
                         PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, typeConverter.getContext(),
                                  benefit) {}
};

/// Helper method to populate the conversion pattern list
void populateQuantumToLLVMConversionPatterns(
    QuantumTypeConverter &typeConveter, OwningRewritePatternList &patterns);

} // namespace quantum
} // namespace mlir

#endif // CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM_H
