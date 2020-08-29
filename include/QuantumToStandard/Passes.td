#ifndef CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD
#define CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD

include "mlir/Pass/PassBase.td"

def QuantumToStandardPass : Pass<"convert-quantum-to-std", "ModuleOp"> {
  let summary = "Convert quantum dialect to standard operations";
  let constructor = "mlir::createConvertQuantumToStandardPass()";
}

#endif // CONVERSION_QUANTUMTOSTANDARD_CONVERTQUANTUMTOSTANDARD
