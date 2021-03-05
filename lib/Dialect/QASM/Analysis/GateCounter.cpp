#include "Dialect/QASM/Analysis/GateCounter.h"
#include "Dialect/QASM/Analysis/Passes.h"
#include "Dialect/QASM/QASMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "PassDetail.h"

using namespace mlir;
using namespace QASM;

namespace {

class GateCountPass : public GateCountPassBase<GateCountPass> {
  void runOnOperation() override;
};

struct GateCounts {
  size_t cnot, u;
};

class GateCountAnalysis {
public:
  GateCountAnalysis(Operation *op) {
    auto module = dyn_cast<ModuleOp>(op);

    module.walk([&](Operation *op) {
      if (auto cnotOp = dyn_cast<ControlledNotOp>(op)) {
        auto func = cnotOp->getParentOfType<FuncOp>();
        addCNOTOp(func);
      }
      if (auto uOp = dyn_cast<SingleQubitRotationOp>(op)) {
        auto func = uOp->getParentOfType<FuncOp>();
        addUOp(func);
      }
      if (auto callOp = dyn_cast<CallOp>(op)) {
        auto func = callOp->getParentOfType<FuncOp>();
        addCallOp(func, callOp);
      }
    });
  }
  void addCNOTOp(FuncOp func) {
    // llvm::errs() << ">> ADDING CX @" << func.sym_name() << '\n';
    this->funcs[func.sym_name()].cnot++;
  }
  void addUOp(FuncOp func) {
    // llvm::errs() << ">> ADDING U @" << func.sym_name() << '\n';
    this->funcs[func.sym_name()].u++;
  }
  void addCallOp(FuncOp func, CallOp call) {
    // llvm::errs() << ">> CALLING " << call.getCallee() << " from "
    //              << func.sym_name() << '\n';
    auto callCounts = funcs[call.getCallee()];
    auto &funcCounts = funcs[func.sym_name()];
    funcCounts.cnot += callCounts.cnot;
    funcCounts.u += callCounts.u;
  }
  GateCounts getGateCounts(llvm::StringRef funcName) { return funcs[funcName]; }

private:
  std::map<llvm::StringRef, GateCounts> funcs;
};

void GateCountPass::runOnOperation() {
  auto analysis = getAnalysis<GateCountAnalysis>();
  llvm::errs() << "// -------- QASM Gate Count analysis --------\n";
  auto gateCounts = analysis.getGateCounts("qasm_main");
  llvm::errs() << "\"qasm_main\": {\n";
  llvm::errs() << "  \"CX\": \"" << gateCounts.cnot << "\",\n";
  llvm::errs() << "  \"U\": \"" << gateCounts.u << "\"\n";
  llvm::errs() << "}\n";
  llvm::errs() << "// ------------------------------------------\n";
}

} // namespace

namespace mlir {
std::unique_ptr<Pass> createQASMGateCountPass() {
  return std::make_unique<GateCountPass>();
}
} // namespace mlir
