#include "TargetOpenQASM/ConvertToOpenQASM.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
using namespace mlir;

int main(int argc, char **argv) {
  registerToOpenQASMTranslation();
  return failed(mlirTranslateMain(argc, argv, "MLIR Quantum Targeting Tool"));
}
