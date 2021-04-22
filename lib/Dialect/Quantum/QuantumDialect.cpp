//===- QuantumDialect.cpp - Quantum dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/Quantum/QuantumTypes.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with qssa
/// operations.
struct QuantumInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Quantum/QuantumOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Quantum/QuantumOpsTypes.cpp.inc"
      >();
  addInterfaces<QuantumInlinerInterface>();
}

//===--- Quantum Types --------------------------------------------------===//
namespace {

static void print(QubitType qubitType, DialectAsmPrinter &os) {
  os << "qubit<";
  if (qubitType.hasStaticSize())
    os << qubitType.getSize();
  else
    os << '?';
  os << ">";
}
static Type parseQubit(DialectAsmParser &parser, MLIRContext *ctx) {
  if (failed(parser.parseLess()))
    return Type();

  int64_t size = -1;
  if (!parser.parseOptionalInteger<int64_t>(size).hasValue() &&
      failed(parser.parseOptionalQuestion())) {
    parser.emitError(parser.getNameLoc(), "expected an integer size or `?`");
    return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  return QubitType::get(ctx, size);
}

} // namespace

bool QubitType::hasStaticSize() const {
  return getSize() != ShapedType::kDynamicSize;
}

bool QubitType::isSingleQubit() const {
  return hasStaticSize() && getSize() == 1;
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/Quantum/QuantumOpsTypes.cpp.inc"

Type QuantumDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return Type();
  Type type;
  generatedTypeParser(getContext(), parser, mnemonic, type);
  return type;
}

/// Print a type registered to this dialect.
void QuantumDialect::printType(Type type, DialectAsmPrinter &os) const {
  (void)generatedTypePrinter(type, os);
}
