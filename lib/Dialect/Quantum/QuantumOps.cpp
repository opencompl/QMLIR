//===- QuantumOps.cpp - Quantum dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

#include "Dialect/Quantum/QuantumOps.h"
#include "TypeDetail.h"

using namespace mlir;
using namespace mlir::quantum;

//==== Folders ==============================================================//
OpFoldResult IDGateOp::fold(ArrayRef<Attribute> operands) { return qinp(); }

OpFoldResult PauliXGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliXGateOp>()) {
    return parent.qinp();
  }
  return *operands.begin();
}

OpFoldResult PauliYGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliYGateOp>()) {
    return parent.qinp();
  }
  return *operands.begin();
}

OpFoldResult PauliZGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliZGateOp>()) {
    return parent.qinp();
  }
  return *operands.begin();
}
OpFoldResult HadamardGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<HadamardGateOp>()) {
    return parent.qinp();
  }
  return *operands.begin();
}

#define GET_OP_CLASSES
#include "Dialect/Quantum/QuantumOps.cpp.inc"
