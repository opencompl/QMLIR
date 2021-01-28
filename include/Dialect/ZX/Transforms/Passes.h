#ifndef TRANSFORM_ZX_ZXREWRITES_PASS
#define TRANSFORM_ZX_ZXREWRITES_PASS

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<FunctionPass> createTransformZXRewritePass();
std::unique_ptr<FunctionPass> createZXCheckSingleUsePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/ZX/Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORM_ZX_ZXREWRITES_PASS
