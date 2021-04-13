add_mlir_dialect(QuantumOps qssa)
add_mlir_doc(QuantumDialect -gen-dialect-doc QuantumDialect Quantum/)
add_mlir_doc(QuantumTypes -gen-op-doc QuantumTypes Quantum/)
add_mlir_doc(QuantumOps -gen-op-doc QuantumOps Quantum/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quantum)
add_public_tablegen_target(MLIRQuantumPassIncGen)

add_subdirectory(Transforms)
