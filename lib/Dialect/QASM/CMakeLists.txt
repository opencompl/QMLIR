add_mlir_dialect_library(MLIRQASM
    QASMDialect.cpp
    QASMTypes.cpp
    QASMOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/QASM

    DEPENDS
    MLIRQASMOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

add_subdirectory(Analysis)
add_subdirectory(Transforms)
