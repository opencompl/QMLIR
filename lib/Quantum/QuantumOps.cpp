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

static ParseResult parseDimAndSymbolList(OpAsmParser &parser,
                                        SmallVectorImpl<Value> &operands,
                                        unsigned &numDims,
                                        OpAsmParser::Delimiter delim = OpAsmParser::Delimiter::Paren) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser.parseOperandList(opInfos, delim))
    return failure();
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // typecheck
  auto indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(opInfos, indexTy, operands))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// AllocateOp
//===----------------------------------------------------------------------===//
static ParseResult parseAllocateOp(OpAsmParser &parser, OperationState &state) {
  QubitType type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // qubit type.
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


//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//
static ParseResult parseConcatOp(OpAsmParser &parser, OperationState &state) {
  // Parse the operands and optional symbol operands, followed by the Op type
  FunctionType opType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(opType))
    return failure();

  if (failed(parser.resolveOperands(operands,
                                    opType.getInputs(),
                                    parser.getCurrentLocation(),
                                    state.operands)))
    return failure();

  state.addTypes(opType.getResults());
  return success();
}

static void print(ConcatOp concatOp, OpAsmPrinter &printer) {
  printer << concatOp.getOperationName() << ' ';

  // print operands
  printer.printOperands(concatOp.getOperands());

  // print optional attributes
  printer.printOptionalAttrDictWithKeyword(concatOp.getAttrs());

  // print the op type
  printer << " : ";
  printer.printFunctionalType(concatOp);
}

static ParseResult verify(ConcatOp concatOp) {
  auto numInputs = concatOp.getODSOperandIndexAndLength(0).second;
  if (numInputs != 2)
    return
      concatOp.emitOpError("expected 2 arguments, provided ")
        << numInputs;
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//
static ParseResult parseSplitOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType inputQubit;
  parser.parseOperand(inputQubit);

  // Parse the size operands and optional symbol operands, followed by
  // the Op type - qubit type followed by a list of qubit types.
  SmallVector<Value, 2> sizeOperands;
  unsigned numSizeOperands;
  QubitType inputType;
  SmallVector<Type, 2> resultTypes;
  if (parseDimAndSymbolList(parser,
                            sizeOperands,
                            numSizeOperands,
                            OpAsmParser::Delimiter::OptionalSquare) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(inputType) ||
      parser.parseArrowTypeList(resultTypes))
    return failure();

  int numDynamicSizeTypes = 0;
  for (auto e: resultTypes) {
    if (auto qubitType = e.cast<QubitType>()) {
      if (!qubitType.hasStaticSize()) {
        numDynamicSizeTypes++;
      }
    } else {
      return parser.emitError(parser.getNameLoc())
           << "Invalid type, expected qubit";
    }
  }

  if (numDynamicSizeTypes != numSizeOperands)
    return parser.emitError(parser.getNameLoc())
           << "Mismatched number of size operands (" << numSizeOperands << ")"
           << " and dynamic-sized qubit arrays (" << numDynamicSizeTypes << ")";

  parser.resolveOperand(inputQubit, inputType, state.operands);
  state.addOperands(sizeOperands);

  state.addTypes(resultTypes);
  return success();
}

static void print(SplitOp splitOp, OpAsmPrinter &printer) {
  printer << splitOp.getOperationName()
          << ' ' << splitOp.getODSOperands(0).front();

  // print size operands
  auto sizeOperands = splitOp.getODSOperands(1);
  if (!sizeOperands.empty()) {
    printer << "[";
    printer.printOperands(sizeOperands);
    printer << "]";
  }

  // print optional attributes
  printer.printOptionalAttrDictWithKeyword(splitOp.getAttrs());

  // print the op type
  printer << " : ";
  printer.printType(splitOp.getODSOperands(0).front().getType());
  printer.printArrowTypeList(splitOp.getResultTypes());
}

static ParseResult verify(SplitOp splitOp) {
  auto numOutputs = splitOp.getODSResultIndexAndLength(0).second;
  if (numOutputs != 2)
    return
      splitOp.emitOpError("expected to split into 2 results, provided ")
        << numOutputs;
  return success();
}

namespace mlir {
namespace quantum {
#define GET_OP_CLASSES
#include "Quantum/QuantumOps.cpp.inc"
} // namespace quantum
} // namespace mlir
