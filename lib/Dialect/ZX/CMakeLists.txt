add_mlir_dialect_library(MLIRZX
    ZXDialect.cpp
    ZXTypes.cpp
    ZXOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/ZX

    DEPENDS
    MLIRZXOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

add_subdirectory(Transforms)
