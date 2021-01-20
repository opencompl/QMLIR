#ifndef TRANSFORM_ZX_ZX
#define TRANSFORM_ZX_ZX

include "mlir/Pass/PassBase.td"

def ZXRewritePass : FunctionPass<"apply-zx-rewrites"> {
  let summary = "Apply ZX rewrite rules";
  let constructor = "mlir::createTransformZXRewritePass()";
}

#endif // TRANSFORM_ZX_ZX
