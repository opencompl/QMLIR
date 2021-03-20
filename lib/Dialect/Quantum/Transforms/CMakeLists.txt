add_mlir_dialect_library(QuantumTransforms
    QuantumRewrites.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Quantum/Transforms

    DEPENDS
    MLIRQuantumTransformsPassIncGen
    MLIRQuantumTransformsRewritesPassIncGen
    )

target_link_libraries(QuantumTransforms PUBLIC MLIRIR)
