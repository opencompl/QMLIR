add_mlir_dialect(QASMOps qasm)
add_mlir_doc(QASMDialect QASMDialect QASM/ -gen-dialect-doc)
add_mlir_doc(QASMTypes QASMTypes QASM/ -gen-op-doc)
add_mlir_doc(QASMOps QASMOps QASM/ -gen-op-doc)

add_subdirectory(Analysis)
add_subdirectory(Transforms)
