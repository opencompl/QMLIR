//===- QuantumDialect.cpp - Quantum dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumTypes.h"
#include "Quantum/QuantumOps.h"

#include "mlir/IR/StandardTypes.h"

// includes
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quantum/QuantumOps.cpp.inc"
  >();
  addTypes<QubitType>();
}

Type QuantumDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Qubit type
  if (keyword == getQubitTypeName()) {
    if (failed(parser.parseLess())) {
      parser.emitError(parser.getNameLoc(), "expected `<`");
      return Type();
    }

    uint64_t size = -1;
    if (!parser.parseOptionalInteger<uint64_t>(size).hasValue() && failed(parser.parseOptionalQuestion())) {
      parser.emitError(parser.getNameLoc(), "expected an integer size or `?`");
      return Type();
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc(), "expected `>`");
      return Type();
    }

    return QubitType::get(parser.getBuilder().getContext(), size);
  }

  parser.emitError(parser.getNameLoc(), "Quantum dialect: unknown type");
  return Type();
}

void QuantumDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<QubitType>()) {
    QubitType qubit = type.cast<QubitType>();

    printer << "qubit<";
    if (qubit.hasStaticSize())
      printer << qubit.getSize();
    else
      printer << "?";
    printer << ">";

    return;
  }
}
