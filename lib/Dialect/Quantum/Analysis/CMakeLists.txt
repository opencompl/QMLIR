add_mlir_dialect_library(QuantumAnalysis
    DepthCompute.cpp
    GateCount.cpp

    # ADDITIONAL_HEADER_DIRS
    # ${PROJECT_SOURCE_DIR}/include/Dialect/Quantum/Analysis

    DEPENDS
    MLIRQuantumPassIncGen
    )

target_link_libraries(QuantumAnalysis PUBLIC MLIRIR)
