add_mlir_dialect(ZXGraphOps zxg)
add_mlir_doc(ZXGraphDialect ZXGraphDialect ZXGraph/ -gen-dialect-doc)
add_mlir_doc(ZXGraphTypes ZXGraphTypes ZXGraph/ -gen-op-doc)
add_mlir_doc(ZXGraphOps ZXGraphOps ZXGraph/ -gen-op-doc)

add_subdirectory(Transforms)
