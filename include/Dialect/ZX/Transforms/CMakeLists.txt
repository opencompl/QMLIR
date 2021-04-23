set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ZXTransforms)
add_public_tablegen_target(MLIRZXTransformsPassIncGen)

set(LLVM_TARGET_DEFINITIONS ZXRewrites.td)
mlir_tablegen(ZXRewrites.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRZXTransformsRewritesPassIncGen)
