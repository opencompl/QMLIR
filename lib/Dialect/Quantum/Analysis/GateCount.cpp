#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace mlir;
using namespace mlir::quantum;

class GateCountPass : public QuantumGateCountPassBase<GateCountPass> {
  void runOnOperation() override;
};

class GateCountAnalysis {
  // gateCounts[funcName][gateName] = #gates
  llvm::StringMap<llvm::StringMap<int64_t>> gateCounts;
  // depths[funcName] = circuit_depth
  llvm::StringMap<int64_t> depths;

  void addGate(FuncOp func, StringRef gateName, int64_t count = 1) {
    gateCounts[func.getName()][gateName] += count;
  }
  void addCall(FuncOp func, StringRef gateCallName) {
    for (auto &childGateCount : gateCounts[gateCallName]) {
      gateCounts[func.getName()][childGateCount.first()] +=
          childGateCount.second;
    }
  }
  void updateDepth(FuncOp func, int64_t depth) {
    depths[func.getName()] = std::max(depths[func.getName()], depth);
  }

public:
  GateCountAnalysis(Operation *op) {
    auto module = dyn_cast<ModuleOp>(op);
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        // compute depth
        if (op->hasAttrOfType<IntegerAttr>("qdepth")) {
          updateDepth(func, op->getAttrOfType<IntegerAttr>("qdepth").getInt());
        }
        // compute gate counts
        if (auto callOp = dyn_cast<CallOp>(op)) {
          addCall(func, callOp.getCallee());
        } else if (isa<CNOTGateOp>(op)) {
          addGate(func, "cx");
        } else if (isa<UniversalRotationGateOp>(op)) {
          addGate(func, "u");
        } else if (auto xGateOp = dyn_cast<PauliXGateOp>(op)) {
          addGate(func, "x", xGateOp.numQubits());
        } else if (auto yGateOp = dyn_cast<PauliYGateOp>(op)) {
          addGate(func, "y", yGateOp.numQubits());
        } else if (auto zGateOp = dyn_cast<PauliZGateOp>(op)) {
          addGate(func, "z", zGateOp.numQubits());
        } else if (auto hGateOp = dyn_cast<HadamardGateOp>(op)) {
          addGate(func, "h", hGateOp.numQubits());
        } else if (auto rxGateOp = dyn_cast<RotateXOp>(op)) {
          addGate(func, "rx", rxGateOp.numQubits());
        } else if (auto ryGateOp = dyn_cast<RotateYOp>(op)) {
          addGate(func, "ry", ryGateOp.numQubits());
        } else if (auto rzGateOp = dyn_cast<RotateZOp>(op)) {
          addGate(func, "rz", rzGateOp.numQubits());
        } else if (auto sGateOp = dyn_cast<PhaseGateOp>(op)) {
          addGate(func, "s", sGateOp.numQubits());
        } else if (auto sdgGateOp = dyn_cast<PhaseDaggerGateOp>(op)) {
          addGate(func, "sdg", sdgGateOp.numQubits());
        } else if (auto tGateOp = dyn_cast<TGateOp>(op)) {
          addGate(func, "t", tGateOp.numQubits());
        } else if (auto tdgGateOp = dyn_cast<TDaggerGateOp>(op)) {
          addGate(func, "tdg", tdgGateOp.numQubits());
        } else if (auto measureOp = dyn_cast<MeasureOp>(op)) {
          addGate(func, "measure",
                  measureOp.qinp().getType().cast<QubitType>().getSize());
        } else if (isa<MeasureQubitOp>(op)) {
          addGate(func, "measure");
        }
      });
    }
  }
  json getStats() const {
    json stats = json::object();
    for (auto &func : gateCounts) {
      json funcStat;
      for (auto &gate : func.getValue()) {
        funcStat[gate.getKey().str()] = gate.getValue();
      }
      stats[func.getKey().str()]["ops"] = funcStat;
      stats[func.getKey().str()]["depth"] = -1;
    }
    for (auto &func : depths) {
      stats[func.getKey().str()]["depth"] = func.getValue();
    }

    return stats;
  }
};

void GateCountPass::runOnOperation() {
  markAllAnalysesPreserved();
  auto analysis = getAnalysis<GateCountAnalysis>();
  llvm::errs() << analysis.getStats().dump(2) << '\n';
}

namespace mlir {

std::unique_ptr<Pass> createQuantumGateCountPass() {
  return std::make_unique<GateCountPass>();
}

} // namespace mlir
