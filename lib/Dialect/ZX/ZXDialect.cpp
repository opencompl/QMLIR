#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "Dialect/ZX/ZXTypes.h"

using namespace mlir;
using namespace mlir::ZX;

void ZXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ZX/ZXOps.cpp.inc"
      >();
  addTypes<WireType>();
}

Type ZXDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Wire type: !zx.wire
  if (keyword == "wire") {
    return WireType::get(parser.getBuilder().getContext());
  }

  parser.emitError(parser.getNameLoc(), "ZX dialect: unknown type");
  return Type();
}

void ZXDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto node = type.cast<WireType>()) {
    printer << "wire";
    return;
  }

  assert(false && "Invalid ZX type given to print");
}
