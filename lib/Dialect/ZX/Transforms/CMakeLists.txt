add_mlir_dialect_library(ZXTransforms
    ZXRewrites.cpp
    CheckSingleUse.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/ZX/Transforms

    DEPENDS
    MLIRZXTransformsPassIncGen
    )

target_link_libraries(ZXTransforms PUBLIC MLIRIR)
