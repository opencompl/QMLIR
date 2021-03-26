#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/ZXGraph/ZXGraphDialect.h"
#include "Dialect/ZXGraph/ZXGraphTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/ZXGraph/ZXGraphOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::ZXGraph;

Type ZXGraphDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return Type();
  Type type;
  generatedTypeParser(getContext(), parser, mnemonic, type);
  return type;
}

/// Print a type registered to this dialect.
void ZXGraphDialect::printType(Type type, DialectAsmPrinter &os) const {
  (void)generatedTypePrinter(type, os);
}
