add_mlir_dialect_library(MLIRQuantum
  QuantumDialect.cpp
  QuantumOps.cpp
  Canonicalizers.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Quantum

  DEPENDS
  MLIRQuantumOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_subdirectory(Analysis)
add_subdirectory(Transforms)
