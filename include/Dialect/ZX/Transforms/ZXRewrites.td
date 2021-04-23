#ifndef ZX_TRANSFORM_ZXREWRITES
#define ZX_TRANSFORM_ZXREWRITES

include "mlir/IR/OpBase.td"
include "Dialect/ZX/ZXOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

// def : Pat<(ZX_ZOp $beta, (ZX_ZOp $alpha, $inp)), (ZX_ZOp (AddFOp $alpha, $beta), $inp)>;

#endif // ZX_TRANSFORM_ZXREWRITES

