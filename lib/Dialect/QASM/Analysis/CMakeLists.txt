add_mlir_dialect_library(QASMAnalysis
    GateCounter.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/QASM/Analysis

    DEPENDS
    MLIRQASMAnalysisPassIncGen
    )

target_link_libraries(QASMAnalysis PUBLIC MLIRIR)
