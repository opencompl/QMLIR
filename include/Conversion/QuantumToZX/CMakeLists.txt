set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QuantumToZXConversion)
add_public_tablegen_target(MLIRQuantumToZXPassIncGen)
