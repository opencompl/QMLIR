set(LLVM_OPTIONAL_SOURCES
  quantum-simulator-wrappers.cpp
  mlir-quantum-simulator.cpp
  )

add_llvm_library(quantum-simulator-wrappers SHARED
  quantum-simulator-wrappers.cpp
) 

add_llvm_tool(mlir-quantum-simulator
  mlir-quantum-simulator.cpp

  DEPENDS
  quantum-simulator-wrappers
)

llvm_update_compile_flags(mlir-quantum-simulator)
