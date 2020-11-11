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
add_llvm_executable(mlir-quantum-target mlir-quantum-target.cpp)

llvm_update_compile_flags(mlir-quantum-target)
target_link_libraries(mlir-quantum-target PRIVATE ${LIBS})

