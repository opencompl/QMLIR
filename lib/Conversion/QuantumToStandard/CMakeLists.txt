add_mlir_dialect_library(QuantumToStandard
  ConvertQuantumToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/QuantumToStandard

  DEPENDS
  MLIRQuantumToStandardPassIncGen
)

target_link_libraries(QuantumToStandard PUBLIC MLIRIR)
