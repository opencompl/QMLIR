add_mlir_dialect_library(QASMToSCF
  ConvertQASMToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/QASMToSCF

  DEPENDS
  MLIRQASMToSCFPassIncGen
)

target_link_libraries(QASMToSCF PUBLIC MLIRIR)
