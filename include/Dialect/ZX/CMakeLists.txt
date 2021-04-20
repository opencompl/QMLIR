add_mlir_dialect(ZXOps zx)
add_mlir_doc(ZXDialect ZXDialect ZX/ -gen-dialect-doc)
add_mlir_doc(ZXTypes ZXTypes ZX/ -gen-op-doc)
add_mlir_doc(ZXOps ZXOps ZX/ -gen-op-doc)

add_subdirectory(Transforms)
