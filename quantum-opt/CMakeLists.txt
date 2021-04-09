get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    nlohmann_json::nlohmann_json
    )
add_llvm_executable(quantum-opt quantum-opt.cpp)

llvm_update_compile_flags(quantum-opt)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
