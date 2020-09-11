set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

set(LLVM_OPTIONAL_SOURCES
  quantum-simulator-wrappers.cpp
  mlir-quantum-simulator.cpp
  simulator-utils.cpp
  matrix-utils.cpp
  )

add_llvm_library(quantum-simulator-wrappers SHARED
  quantum-simulator-wrappers.cpp
  simulator-utils.cpp
  matrix-utils.cpp
  )

add_llvm_tool(mlir-quantum-simulator
  mlir-quantum-simulator.cpp

  DEPENDS
  quantum-simulator-wrappers
  )

llvm_update_compile_flags(mlir-quantum-simulator)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-quantum-simulator PRIVATE
  ${dialect_libs}
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  )
