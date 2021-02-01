add_mlir_dialect_library(MLIRZXGraph
    ZXGraphDialect.cpp
    ZXGraphTypes.cpp
    ZXGraphOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/ZXGraph

    DEPENDS
    MLIRZXGraphOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

add_subdirectory(Transforms)
