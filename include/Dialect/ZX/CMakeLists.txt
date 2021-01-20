add_mlir_dialect(ZXOps zx)
add_mlir_doc(ZXDialect -gen-dialect-doc ZXDialect ZX/)
add_mlir_doc(ZXTypes -gen-op-doc ZXTypes ZX/)
add_mlir_doc(ZXOps -gen-op-doc ZXOps ZX/)

add_subdirectory(Transforms)
