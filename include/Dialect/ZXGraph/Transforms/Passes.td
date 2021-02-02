#ifndef TRANSFORM_ZXGRAPH_ZX
#define TRANSFORM_ZXGRAPH_ZX

include "mlir/Pass/PassBase.td"

def ZXGraphRewritePass : FunctionPass<"zxg-apply-rewrites"> {
  let summary = "Apply ZXGraph rewrite rules";
  let constructor = "mlir::createTransformZXGraphRewritePass()";
}

def ZXGraphCanonicalizePass : FunctionPass<"zxg-canonicalize-blocks"> {
  let summary = "Canonicalize blocks. Move ZXGraph wires to the bottom.";
  let constructor = "mlir::createTransformZXGraphCanonicalizePass()";
}

#endif // TRANSFORM_ZXGRAPH_ZX
