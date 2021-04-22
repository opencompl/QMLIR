add_mlir_dialect_library(QuantumToLLVM
    ConvertQuantumToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Conversion/QuantumToLLVM

  DEPENDS
  MLIRQuantumToLLVMPassIncGen
)

target_link_libraries(QuantumToLLVM PUBLIC MLIRIR)
