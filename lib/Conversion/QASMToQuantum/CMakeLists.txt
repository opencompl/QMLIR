add_mlir_dialect_library(QASMToQuantum
  ConvertQASMToQuantum.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/QASMToQuantum

  DEPENDS
  MLIRQASMToQuantumPassIncGen
)

target_link_libraries(QASMToQuantum PUBLIC MLIRIR)
