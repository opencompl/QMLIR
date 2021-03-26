#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "Dialect/ZX/ZXTypes.h"

using namespace mlir;
using namespace mlir::ZX;

namespace {
/// This class defines the interface for handling inlining with qssa
/// operations.
struct ZXInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void ZXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ZX/ZXOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ZX/ZXOpsTypes.cpp.inc"
      >();
  addInterfaces<ZXInlinerInterface>();
}
