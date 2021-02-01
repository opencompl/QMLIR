add_mlir_dialect(ZXGraphOps zxg)
add_mlir_doc(ZXGraphDialect -gen-dialect-doc ZXGraphDialect ZXGraph/)
add_mlir_doc(ZXGraphTypes -gen-op-doc ZXGraphTypes ZXGraph/)
add_mlir_doc(ZXGraphOps -gen-op-doc ZXGraphOps ZXGraph/)

add_subdirectory(Transforms)
