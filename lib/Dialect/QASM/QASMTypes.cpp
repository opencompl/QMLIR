#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/QASM/QASMOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::QASM;

Type QASMDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return Type();
  Type type;
  generatedTypeParser(getContext(), parser, mnemonic, type);
  return type;
}

/// Print a type registered to this dialect.
void QASMDialect::printType(Type type, DialectAsmPrinter &os) const {
  (void)generatedTypePrinter(type, os);
}
