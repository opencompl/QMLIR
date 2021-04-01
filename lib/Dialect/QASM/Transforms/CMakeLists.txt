add_mlir_dialect_library(QASMTransforms
  MakeGatesOpaque.cpp
  MakeGatesTransparent.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/QASM/Transforms

  DEPENDS
  MLIRQASMTransformsPassIncGen
  )

target_link_libraries(QASMTransforms PUBLIC MLIRIR)
