add_mlir_dialect_library(QuantumToZX
  ConvertQuantumToZX.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/QuantumToZX

  DEPENDS
  MLIRQuantumToZXPassIncGen
)

target_link_libraries(QuantumToZX PUBLIC MLIRIR)
