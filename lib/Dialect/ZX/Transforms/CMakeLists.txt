add_mlir_dialect_library(ZXTransforms
  ZXTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/ZX/Transforms

  DEPENDS
  MLIRZXTransformsPassIncGen
)

target_link_libraries(ZXTransforms PUBLIC MLIRIR)
