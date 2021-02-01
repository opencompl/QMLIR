#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/ZXGraph/ZXGraphDialect.h"
#include "Dialect/ZXGraph/ZXGraphOps.h"
#include "Dialect/ZXGraph/ZXGraphTypes.h"

using namespace mlir;
using namespace mlir::ZXGraph;

void ZXGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ZXGraph/ZXGraphOps.cpp.inc"
      >();
  addTypes<NodeType>();
}

Type ZXGraphDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Wire type: !zx.wire
  if (keyword == "node") {
    return NodeType::get(parser.getBuilder().getContext());
  }

  parser.emitError(parser.getNameLoc(), "ZXGraph dialect: unknown type");
  return Type();
}

void ZXGraphDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto node = type.cast<NodeType>()) {
    printer << "node";
    return;
  }

  assert(false && "Invalid ZXGraph type given to print");
}
