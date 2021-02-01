add_mlir_dialect_library(ZXGraphTransforms
    ZXGraphRewrites.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/ZXGraph/Transforms

    DEPENDS
    MLIRZXGraphTransformsPassIncGen
    )

target_link_libraries(ZXGraphTransforms PUBLIC MLIRIR)
