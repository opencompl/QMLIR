#ifndef TRANSFORM_ZXGRAPH_ZX
#define TRANSFORM_ZXGRAPH_ZX

include "mlir/Pass/PassBase.td"

def ZXGraphRewritePass : FunctionPass<"apply-zxg-rewrites"> {
  let summary = "Apply ZXGraph rewrite rules";
  let constructor = "mlir::createTransformZXGraphRewritePass()";
}

#endif // TRANSFORM_ZXGRAPH_ZX
