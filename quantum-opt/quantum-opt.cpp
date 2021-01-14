//===- quantum-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

#include "Conversion/QuantumToStandard/Passes.h"
#include "Dialect/QASM/ZXDialect.h"
#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/ZX/ZXDialect.h"

using namespace mlir;

void registerQuantumPasses() { registerQuantumConversionPasses(); }

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<quantum::QuantumDialect>();
  registry.insert<QASM::QASMDialect>();
  registry.insert<ZX::ZXDialect>();

  registerAllPasses();
  registerQuantumPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "Quantum dialect driver\n", registry));
}
