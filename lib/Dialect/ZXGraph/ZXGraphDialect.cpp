#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/ZXGraph/ZXGraphDialect.h"
#include "Dialect/ZXGraph/ZXGraphOps.h"
#include "Dialect/ZXGraph/ZXGraphTypes.h"

using namespace mlir;
using namespace mlir::ZXGraph;

namespace {
/// This class defines the interface for handling inlining with qssa
/// operations.
struct ZXGraphInlinerInterface : public DialectInlinerInterface {
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

void ZXGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ZXGraph/ZXGraphOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ZXGraph/ZXGraphOpsTypes.cpp.inc"
      >();

  addInterfaces<ZXGraphInlinerInterface>();
}
