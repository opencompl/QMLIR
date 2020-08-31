//===- QuantumOps.cpp - Quantum dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quantum/QuantumOps.h"
#include "Quantum/QuantumDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace quantum;

//===----------------------------------------------------------------------===//
// AllocateOp
//===----------------------------------------------------------------------===//
static ParseResult parseDimAndSymbolList(OpAsmParser &parser,
                                        SmallVectorImpl<Value> &operands,
                                        unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser.parseOperandList(opInfos, OpAsmParser::Delimiter::Paren))
    return failure();
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // typecheck
  auto indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(opInfos, indexTy, operands))
    return failure();

  return success();
}

static ParseResult parseAllocateOp(OpAsmParser &parser, OperationState &state) {
  QubitType type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, state.operands, numDimOperands) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(type))
    return failure();

  if (numDimOperands > 1)
    return parser.emitError(parser.getNameLoc())
           << "Too many dynamic dimension operands provided";

  if (type.hasStaticSize() && numDimOperands != 0)
    return parser.emitError(parser.getNameLoc())
           << "Too many dynamic dimension operands provided";
  
  if (!type.hasStaticSize() && numDimOperands == 0)
    return parser.emitError(parser.getNameLoc())
           << "Dynamic dimension operand not provided";
  
  
  state.types.push_back(type);
  return success();
}

static void print(quantum::AllocateOp allocateOp, OpAsmPrinter &printer) {
  printer << allocateOp.getOperationName();
  
  // print size operands
  printer << "(";
  printer.printOperands(allocateOp.dynamicsize());
  printer << ")";

  // print optional attributes
  printer.printOptionalAttrDictWithKeyword(allocateOp.getAttrs());

  // print the qubit type
  printer << " : ";
  printer.printType(allocateOp.qout().getType());
}

namespace mlir {
namespace quantum {
#define GET_OP_CLASSES
#include "Quantum/QuantumOps.cpp.inc"
} // namespace quantum
} // namespace mlir
