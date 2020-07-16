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

QuantumDialect::QuantumDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "Quantum/QuantumOps.cpp.inc"
  >();
  addTypes<QubitType>();
}

mlir::Type QuantumDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  llvm::SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseKeyword(&keyword))) {
    return Type();
  }

  if (keyword == "qubit") {
    unsigned size;
    if (failed(parser.parseLess())
      || failed(parser.parseInteger(size))
      || failed(parser.parseGreater()))
      return Type();
    return QubitType::get(parser.getBuilder().getContext(), size);
  }

  parser.emitError(loc, "Quantum dialect: Invalid type");
  return Type();
}

void QuantumDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<QubitType>()) {
    printer << "qubit<" << type.cast<QubitType>().getSize() << ">";
  }
}
