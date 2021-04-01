#ifndef QASM_TRANSFORMS_PASSES_TD
#define QASM_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def QASMMakeGatesOpaquePass : FunctionPass<"qasm-make-gates-opaque"> {
  let summary = "Convert std.call to qasm.gate opaque instructions";
  let constructor = "mlir::createQASMMakeGatesOpaquePass()";
  let options = [
    ListOption<"gates", "gates", "std::string", "List of gates to convert",
      "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}
def QASMMakeGatesTransparentPass : FunctionPass<"qasm-make-gates-transparent"> {
  let summary = "Convert qasm.gate opaque instructions to std.call";
  let constructor = "mlir::createQASMMakeGatesTransparentPass()";
  let options = [
    ListOption<"gates", "gates", "std::string", "List of gates to convert",
      "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

#endif // QASM_TRANSFORMS_PASSES_TD
