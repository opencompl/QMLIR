#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/ZX/ZXOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::ZX;

Type ZXDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return Type();
  Type type;
  generatedTypeParser(getContext(), parser, mnemonic, type);
  return type;
}

/// Print a type registered to this dialect.
void ZXDialect::printType(Type type, DialectAsmPrinter &os) const {
  (void)generatedTypePrinter(type, os);
}
