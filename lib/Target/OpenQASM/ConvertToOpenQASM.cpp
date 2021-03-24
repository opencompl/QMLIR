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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"

#include "llvm/Support/ToolOutputFile.h"
#include <string>
#include <vector>

#include "Dialect/QASM/QASMDialect.h"
#include "Dialect/QASM/QASMOps.h"
#include "Target/OpenQASM/ConvertToOpenQASM.h"

using namespace mlir;
using namespace QASM;

namespace {

class QASMTranslation {
  ModuleOp module;
  raw_ostream &output;

  DenseMap<Value, std::string> scope;

  // Helpers
  void addQubit(Value qubit);
  void addCreg(Value qubit);
  StringRef lookupScope(Value v) { return scope[v]; }
  void splitArguments(ArrayRef<Value> args, SmallVector<Value> &params,
                      SmallVector<Value> &qubits);

  std::string flattenExpr(Value val);
  std::string flattenExprBraced(Value val);

  template <typename T>
  WalkResult runPrinters(Operation *op, StringRef prefix = "") {
    if (auto opp = dyn_cast<T>(op)) {
      output << prefix;
      if (failed(print(opp)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  template <typename T, typename V, typename... Ts>
  WalkResult runPrinters(Operation *op, StringRef prefix = "") {
    if (auto opp = dyn_cast<T>(op)) {
      output << prefix;
      if (failed(print(opp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    return runPrinters<V, Ts...>(op, prefix);
  }

  // Op printers
  LogicalResult print(AllocateOp op);
  LogicalResult print(memref::AllocOp op);
  LogicalResult print(ResetOp op);
  LogicalResult print(MeasureOp op);
  LogicalResult print(BarrierOp op);
  LogicalResult print(ControlledNotOp op);
  LogicalResult print(SingleQubitRotationOp op);
  LogicalResult print(CallOp op);
  LogicalResult print(GlobalPhaseGateOp op);
  LogicalResult print(IfOp op);

  // Function translations
  LogicalResult translateGate(FuncOp gateFunc);
  LogicalResult translateOpaque(FuncOp gateFunc);
  LogicalResult translateMain(FuncOp mainFunc);

public:
  QASMTranslation(ModuleOp op, raw_ostream &output)
      : module(op), output(output) {}
  /// Convert the MLIR module to OpenQASM
  LogicalResult translate();
};

void QASMTranslation::addQubit(Value qubit) {
  int idx = scope.size();
  scope[qubit] = "q" + std::to_string(idx);
}
void QASMTranslation::addCreg(Value creg) {
  int idx = scope.size();
  scope[creg] = "c" + std::to_string(idx);
}

void QASMTranslation::splitArguments(ArrayRef<Value> args,
                                     SmallVector<Value> &params,
                                     SmallVector<Value> &qubits) {
  for (auto arg : args) {
    if (arg.getType().isa<QubitType>()) {
      qubits.push_back(arg);
    }
    if (arg.getType().isa<FloatType, IntegerType>()) {
      params.push_back(arg);
    }
  }
}

std::string QASMTranslation::flattenExprBraced(Value val) {
  return "(" + flattenExpr(val) + ")";
}
std::string QASMTranslation::flattenExpr(Value val) {
  // check if it is a variable in scope
  if (scope.find(val) != scope.end()) {
    return scope[val];
  }

  // If constant, return as-is
  if (auto constOp = val.getDefiningOp<ConstantOp>()) {
    auto attr = constOp.getValue();
    if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
      return std::to_string(intAttr.getInt());
    }
    if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
      return std::to_string(floatAttr.getValueAsDouble());
    }
    assert(false && "invalid constant");
  }
  if (auto piOp = val.getDefiningOp<PIOp>()) {
    return std::to_string(double(M_PI));
  }

  // Float ops
  if (auto addFOp = val.getDefiningOp<AddFOp>()) {
    return flattenExprBraced(addFOp.lhs()) + " + " +
           flattenExprBraced(addFOp.rhs());
  }
  if (auto subFOp = val.getDefiningOp<SubFOp>()) {
    return flattenExprBraced(subFOp.lhs()) + " - " +
           flattenExprBraced(subFOp.rhs());
  }
  if (auto mulFOp = val.getDefiningOp<MulFOp>()) {
    return flattenExprBraced(mulFOp.lhs()) + " * " +
           flattenExprBraced(mulFOp.rhs());
  }
  if (auto divFOp = val.getDefiningOp<DivFOp>()) {
    return flattenExprBraced(divFOp.lhs()) + " / " +
           flattenExprBraced(divFOp.rhs());
  }
  if (auto negFOp = val.getDefiningOp<NegFOp>()) {
    return "-" + flattenExprBraced(negFOp.operand());
  }

  // Int ops
  if (auto addIOp = val.getDefiningOp<AddIOp>()) {
    return flattenExprBraced(addIOp.lhs()) + " + " +
           flattenExprBraced(addIOp.rhs());
  }
  if (auto subIOp = val.getDefiningOp<SubIOp>()) {
    return flattenExprBraced(subIOp.lhs()) + " - " +
           flattenExprBraced(subIOp.rhs());
  }
  if (auto mulIOp = val.getDefiningOp<MulIOp>()) {
    return flattenExprBraced(mulIOp.lhs()) + " * " +
           flattenExprBraced(mulIOp.rhs());
  }

  // casting ops
  if (auto sitofp = val.getDefiningOp<SIToFPOp>()) {
    return flattenExpr(sitofp.in());
  }

  assert(false && "invalid expression op");
}

LogicalResult QASMTranslation::print(AllocateOp op) {
  addQubit(op.qout());
  output << "qreg " << lookupScope(op.qout()) << "[1];\n";
  return success();
}
LogicalResult QASMTranslation::print(memref::AllocOp op) {
  auto memrefType = op.memref().getType().cast<MemRefType>();
  addCreg(op.memref());
  output << "creg " << lookupScope(op.memref()) << "["
         << memrefType.getShape()[0] << "];\n";
  return success();
}
LogicalResult QASMTranslation::print(ResetOp op) {
  output << "reset " << lookupScope(op.qinp()) << ";\n";
  return success();
}
LogicalResult QASMTranslation::print(MeasureOp op) {
  auto users = op->getUsers();
  if (users.empty()) // ignore
    return success();
  auto *firstUse = *users.begin();

  if (auto storeOp = dyn_cast<memref::StoreOp>(firstUse)) {
    auto creg = lookupScope(storeOp.memref());
    auto index = flattenExpr(storeOp.indices()[0]);
    if (!creg.empty()) {
      output << "measure " << lookupScope(op.qinp()) << " -> " << creg << "["
             << index << "];\n";
      return success();
    }
  }
  if (auto storeOp = dyn_cast<AffineStoreOp>(firstUse)) {
    auto creg = lookupScope(storeOp.memref());
    std::string index;
    if (!storeOp.indices().empty())
      index = flattenExpr(storeOp.indices()[0]);
    else
      index = std::to_string(
          storeOp.getAffineMapAttr().getValue().getSingleConstantResult());
    if (!creg.empty()) {
      output << "measure " << lookupScope(op.qinp()) << " -> " << creg << "["
             << index << "];\n";
      return success();
    }
  }

  emitError(op->getLoc()) << "Unable to validate measure op: invalid first use "
                             "(not storing into a creg)";
  emitError(firstUse->getLoc()) << "First use here";
  return failure();
}
LogicalResult QASMTranslation::print(BarrierOp op) {
  output << "barrier " << lookupScope(op.qinp()) << ";\n";
  return success();
}
LogicalResult QASMTranslation::print(ControlledNotOp op) {
  output << "CX " << lookupScope(op.qinp0()) << ", " << lookupScope(op.qinp1())
         << ";\n";
  return success();
}
LogicalResult QASMTranslation::print(SingleQubitRotationOp op) {
  output << "U(" << flattenExpr(op.theta()) << ", " << flattenExpr(op.phi())
         << ", " << flattenExpr(op.lambda()) << ")";
  output << " " << lookupScope(op.qinp()) << ";\n";
  return success();
}
LogicalResult QASMTranslation::print(CallOp op) {
  SmallVector<Value> args, params, qargs;
  for (auto arg : op.getArgOperands())
    args.push_back(arg);
  splitArguments(args, params, qargs);
  output << op.getCallee() << " (";
  llvm::interleaveComma(params, output,
                        [&](Value arg) { output << flattenExpr(arg); });
  output << ") ";
  llvm::interleaveComma(qargs, output,
                        [&](Value arg) { output << lookupScope(arg); });
  output << ";\n";
  return success();
}
LogicalResult QASMTranslation::print(GlobalPhaseGateOp op) {
  output << "gphase(" << flattenExpr(op.gamma()) << ");\n";
  return success();
}
LogicalResult QASMTranslation::print(IfOp op) {
  std::string ifCond = ("if(" + lookupScope(op.creg()) +
                        " == " + std::to_string(op.value()) + ") ")
                           .str();
  for (auto &sub : op.ifBlock().getOps()) {
    auto res = runPrinters<SingleQubitRotationOp, ControlledNotOp, MeasureOp,
                           ResetOp, CallOp>(&sub, ifCond);
    if (res.wasInterrupted())
      return failure();
  }
  return success();
}

LogicalResult QASMTranslation::translateGate(FuncOp gateFunc) {
  output << "gate " << gateFunc.getName() << " ";
  SmallVector<Value> params, qargs;
  auto funcArgs = gateFunc.getBody().getArguments();
  splitArguments({funcArgs.begin(), funcArgs.end()}, params, qargs);

  DenseMap<Value, std::string> names;

  if (!params.empty())
    output << "(";
  for (auto arg : llvm::enumerate(params)) {
    names[arg.value()] = "p" + std::to_string(arg.index());
    if (arg.index() > 0)
      output << ", ";
    output << names[arg.value()];
  }
  if (!params.empty())
    output << ") ";

  for (auto arg : llvm::enumerate(qargs)) {
    names[arg.value()] = "q" + std::to_string(arg.index());
    if (arg.index() > 0)
      output << ", ";
    output << names[arg.value()];
  }

  // use local scope
  swap(names, scope);
  output << " {\n";

  WalkResult result = gateFunc.walk([&](Operation *op) {
    return runPrinters<ControlledNotOp, SingleQubitRotationOp, CallOp,
                       BarrierOp>(op);
  });
  if (result.wasInterrupted())
    return failure();

  // restore scope
  swap(names, scope);
  output << "}\n";

  return success();
}

LogicalResult QASMTranslation::translateOpaque(FuncOp gateFunc) {
  SmallVector<std::string> params, qubits;
  for (auto arg : llvm::enumerate(gateFunc.getType().getInputs())) {
    if (arg.value().isa<QubitType>())
      qubits.push_back("q" + std::to_string(arg.index()));
    else
      params.push_back("p" + std::to_string(arg.index()));
  }

  output << "opaque " << gateFunc.getName() << "(";
  llvm::interleaveComma(params, output);
  output << ") ";
  llvm::interleaveComma(qubits, output);
  output << ";\n";

  return success();
}

LogicalResult QASMTranslation::translateMain(FuncOp mainFunc) {
  WalkResult result = mainFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto opp = dyn_cast<IfOp>(op)) {
      if (failed(print(opp)))
        return WalkResult::interrupt();
      return WalkResult::skip();
    }
    return runPrinters<AllocateOp, memref::AllocOp, ResetOp, MeasureOp,
                       BarrierOp, ControlledNotOp, SingleQubitRotationOp,
                       CallOp, GlobalPhaseGateOp>(op);
  });

  if (result.wasInterrupted())
    return failure();
  return success();
}

LogicalResult QASMTranslation::translate() {
  // target headers
  output << "OPENQASM 2.0;\n";
  output << "include \"qelib1.inc\";\n";
  output << "\n";

  for (auto func : module.getOps<FuncOp>()) {
    // ignore standard gates (already present in qelib1.inc)
    if (func->hasAttr("qasm.stdgate"))
      continue;

    // print gate definition
    if (func->hasAttr("qasm.gate")) {
      if (func.isDeclaration()) {
        if (failed(translateOpaque(func)))
          return failure();
      } else {
        if (failed(translateGate(func)))
          return failure();
      }
    }

    // print main function
    if (func->hasAttr("qasm.main"))
      if (failed(translateMain(func)))
        return failure();
  }
  return success();
}

} // namespace

namespace mlir {
void registerToOpenQASMTranslation() {
  [[maybe_unused]] TranslateFromMLIRRegistration registration(
      "mlir-to-openqasm",
      [](ModuleOp module, raw_ostream &output) {
        return QASMTranslation(module, output).translate();
      },
      [](DialectRegistry &registry) {
        registry.insert<QASM::QASMDialect, StandardOpsDialect,
                        memref::MemRefDialect, AffineDialect>();
      });
}
} // namespace mlir
