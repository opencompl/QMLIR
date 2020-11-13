get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
    ${dialect_libs}
    MLIRQuantum
    MLIRAnalysis
    MLIREDSC
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRSupport
    )
add_llvm_executable(quantum-translate quantum-translate.cpp)

llvm_update_compile_flags(quantum-translate)
target_link_libraries(quantum-translate PRIVATE ${LIBS})

