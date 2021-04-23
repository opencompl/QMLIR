#ifndef CONVERSION_ZXTOQUANTUM_PASSES_TD
#define CONVERSION_ZXTOQUANTUM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ZXToQuantumPass : Pass<"convert-zx-to-quantum", "ModuleOp"> {
  let summary = "Convert zx graph to qssa circuit";
  let constructor = "mlir::createZXToQuantumPass()";
  let dependentDialects = ["ZX::ZXDialect",
                           "quantum::QuantumDialect",
                           "StandardOpsDialect"];
}

#endif // CONVERSION_ZXTOQUANTUM_PASSES_TD
