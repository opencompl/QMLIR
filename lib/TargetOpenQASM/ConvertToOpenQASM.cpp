//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "TargetOpenQASM/ConvertToOpenQASM.h"
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Translation.h"

#include "llvm/Support/ToolOutputFile.h"
#include <map>
#include <string>
#include <vector>
using namespace mlir;
using namespace quantum;

/// Global registry of qubits
/// Does naive allocation to array and maintains references
class QubitRegistry {
  using QubitRef = std::pair<std::string, uint64_t>;
  uint64_t numQubitArrays;
  std::vector<std::pair<Value, std::string>> varNames;
  std::vector<std::pair<Value, std::vector<QubitRef>>> qubits;

public:
  QubitRegistry() : numQubitArrays(0), varNames(), qubits() {}
  std::string getName(Value v) {
    for (auto &var: varNames) {
      auto vv = var.first;
      auto name = var.second;
      if (v == vv)
        return name;
    }
    return "";
  }
  std::string addQubits(Value v) {
    auto name = std::string("qs") + std::to_string(numQubitArrays);
    varNames.emplace_back(v, name);
    numQubitArrays++;

    std::vector<QubitRef> refs;
    auto size = v.getType().cast<QubitType>().getSize();
    for (auto i = 0; i < size; i++) {
      refs.emplace_back(name, i);
    }
    qubits.emplace_back(v, refs);

    return varNames.back().second;
  }
  void update(Value oldv, Value newv) {
    qubits.push_back(std::make_pair(newv, getQubits(oldv)));
  }

  std::vector<QubitRef> getQubits(Value v) {
    for (auto &qubit : qubits) {
      auto vv = qubit.first;
      auto qref = qubit.second;
      if (v == vv) {
        return qref;
      }
    }
    assert(false && "invalid state: qubit slice not found");
  }
};

/// Convert MLIR to OpenQASM
static void ModuleToOpenQASM(ModuleOp op, raw_ostream &output) {
  // target headers
  output << "OPENQASM 3;\n";
  output << "include \"stdgates.inc\"\n";
  output << "\n";

  // conversion
  QubitRegistry qubitRegistry;

  op.walk([&](AllocateOp allocateOp) {
    auto result = allocateOp.getResult();
    auto qubit = result.getType().cast<QubitType>();
    if (!qubit.hasStaticSize()) {
      allocateOp.emitError("Cannot use dynamic sizes in QASM");
      return WalkResult::interrupt();
    }

    auto name = qubitRegistry.addQubits(result);
    output << "qubit[" << qubit.getSize() << "] " << name << ";\n";

    return WalkResult::advance();
  });

  op.walk([&](PauliXGateOp pauliXGateOp) {
    auto argument = pauliXGateOp.qinp();
    auto qubits = qubitRegistry.getQubits(argument);
    for (auto &qubit : qubits) {
      auto &name = qubit.first;
      auto &index = qubit.second;
      output << "x " << name << '[' << index << "];\n";
    }
    qubitRegistry.update(argument, pauliXGateOp.getResult());

    return WalkResult::advance();
  });

  uint64_t numBitArrays = 0;
  op.walk([&](MeasureOp measureOp) {
    auto arg = measureOp.qinp();
    auto size = arg.getType().cast<QubitType>().getSize();
    std::string resName = "bs" + std::to_string(numBitArrays);
    output << "bits[" << size << "] " << resName << ";\n";
    numBitArrays++;

    int i = 0;
    for (auto &e : qubitRegistry.getQubits(arg)) {
      auto name = e.first;
      auto index = e.second;
      output << resName << '[' << i << "] = measure " << name << "[" << index
             << "];\n";
      i++;
    }

    return WalkResult::advance();
  });
}

namespace mlir {
void registerToOpenQASMTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-openqasm",
      [](ModuleOp module, raw_ostream &output) {
        ModuleToOpenQASM(module, output);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<LLVM::LLVMDialect, quantum::QuantumDialect,
                        StandardOpsDialect>();
      });
}
} // namespace mlir
