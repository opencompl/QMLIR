#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "Target/OpenQASM/ConvertToOpenQASM.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();
  registerToOpenQASMTranslation();
  return failed(mlirTranslateMain(argc, argv, "MLIR Quantum Translation Tool"));
}
