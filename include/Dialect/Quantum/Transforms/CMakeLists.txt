set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QuantumTransforms)
add_public_tablegen_target(MLIRQuantumTransformsPassIncGen)

set(LLVM_TARGET_DEFINITIONS QuantumRewrites.td)
mlir_tablegen(QuantumRewrites.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRQuantumTransformsRewritesPassIncGen)
