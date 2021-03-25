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
LogicalResult SplitOp::fold(ArrayRef<Attribute> operands,
                            SmallVectorImpl<OpFoldResult> &results) {
  if (auto parentConcatOp = qinp().getDefiningOp<ConcatOp>()) {
    if (parentConcatOp.qout() == qinp()) {
      results.assign(parentConcatOp.qinps().begin(),
                     parentConcatOp.qinps().end());
      return success();
    }
  }
  return failure();
}

OpFoldResult ConcatOp::fold(ArrayRef<Attribute> operands) {
  if (auto parentSplitOp = qinps()[0].getDefiningOp<SplitOp>()) {
    if (parentSplitOp.qouts() == qinps())
      return parentSplitOp.qinp();
  }
  return nullptr;
}

OpFoldResult IDGateOp::fold(ArrayRef<Attribute> operands) { return qinp(); }

OpFoldResult PauliXGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliXGateOp>()) {
    return parent.qinp();
  }
  return nullptr;
}

OpFoldResult PauliYGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliYGateOp>()) {
    return parent.qinp();
  }
  return nullptr;
}

OpFoldResult PauliZGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<PauliZGateOp>()) {
    return parent.qinp();
  }
  return nullptr;
}
OpFoldResult HadamardGateOp::fold(ArrayRef<Attribute> operands) {
  if (auto parent = qinp().getDefiningOp<HadamardGateOp>()) {
    return parent.qinp();
  }
  return nullptr;
}

LogicalResult
quantum::CNOTGateOp::fold(ArrayRef<Attribute> operands,
                          SmallVectorImpl<OpFoldResult> &results) {
  if (auto parentCNOT = qinp_cont().getDefiningOp<CNOTGateOp>()) {
    if (parentCNOT.qout_cont() == qinp_cont() &&
        parentCNOT.qout_targ() == qinp_targ()) {
      results.push_back(parentCNOT.qinp_cont());
      results.push_back(parentCNOT.qinp_targ());
      return success();
    }
  }
  return failure();
}

#define GET_OP_CLASSES
#include "Dialect/Quantum/QuantumOps.cpp.inc"
