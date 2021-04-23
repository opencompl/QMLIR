add_mlir_dialect_library(ZXToQuantum
  ConvertZXToQuantum.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/ZXToQuantum

  DEPENDS
  MLIRZXToQuantumPassIncGen
)

target_link_libraries(ZXToQuantum PUBLIC MLIRIR)
