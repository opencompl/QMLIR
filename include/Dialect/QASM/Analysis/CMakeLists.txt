set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QASMAnalysis)
add_public_tablegen_target(MLIRQASMAnalysisPassIncGen)
