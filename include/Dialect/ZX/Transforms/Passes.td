#ifndef TRANSFORM_ZX_ZX
#define TRANSFORM_ZX_ZX

include "mlir/Pass/PassBase.td"

def ZXRewritePass : FunctionPass<"apply-zx-rewrites"> {
  let summary = "Apply ZX rewrite rules";
  let constructor = "mlir::createTransformZXRewritePass()";
}

def ZXCheckSingleUsePass : FunctionPass<"zx-check-single-use"> {
  let summary = "Verify that all wires are only used once";
  let constructor = "mlir::createZXCheckSingleUsePass()";
}

#endif // TRANSFORM_ZX_ZX
