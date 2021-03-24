add_mlir_translation_library(QASMToOpenQASM
    ConvertToOpenQASM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/OpenQASM

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRQASM
    )

target_link_libraries(QASMToOpenQASM PUBLIC MLIRIR)
