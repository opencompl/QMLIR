#ifndef TRANSFORM_ZX_ZXREWRITES
#define TRANSFORM_ZX_ZXREWRITES

// mlir-tblgen --gen-rewriters -I {PATH_TO_LLVM}/mlir/include -I ${ROOT}/include ${ROOT}/include/Dialect/ZX/Transforms/ZXRewrites.td

include "mlir/IR/OpBase.td"
include "Dialect/ZX/ZXOps.td"

// def : Pat<(ZX_ZOp $angle, $input)__0, (ZX_ZOp $angle, $input)>;

#endif // TRANSFORM_ZX_ZXREWRITES

