#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/QASM/QASMOps.h"
#include "Dialect/QASM/QASMTypes.h"
#include "Dialect/QASM/ZXDialect.h"

using namespace mlir;
using namespace mlir::QASM;

void QASMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/QASM/QASMOps.cpp.inc"
      >();
  addTypes<QubitType>();
}

Type QASMDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Qubit type: !qasm.qubit<n>
  if (keyword == getQubitTypeName()) {
    uint64_t size;
    if (failed(parser.parseLess()) || failed(parser.parseInteger(size)) ||
        failed(parser.parseGreater())) {
      parser.emitError(parser.getCurrentLocation(),
                       "Invalid syntax for qubit type.");
      return Type();
    }
    return QubitType::get(parser.getBuilder().getContext(), size);
  }

  parser.emitError(parser.getNameLoc(), "QASM dialect: unknown type");
  return Type();
}

void QASMDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto qubit = type.cast<QubitType>()) {
    printer << getQubitTypeName() << "<" << qubit.getSize() << ">";
    return;
  }

  assert(false && "Invalid QASM type given to print");
}
