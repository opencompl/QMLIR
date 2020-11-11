add_mlir_dialect_library(QuantumToOpenQASM
    ConvertToOpenQASM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TargetOpenQASM

    LINK_LIBS PUBLIC
    MLIRIR
    )

target_link_libraries(QuantumToOpenQASM PUBLIC MLIRIR)
