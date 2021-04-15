add_mlir_dialect_library(QuantumToQASM
  ConvertQuantumToQASM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/QuantumToQASM

  DEPENDS
  MLIRQuantumToQASMPassIncGen
)

target_link_libraries(QuantumToQASM PUBLIC MLIRIR)
