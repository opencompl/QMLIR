//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM
#define CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM

include "mlir/Pass/PassBase.td"

def QuantumToLLVMPass : Pass<"convert-qssa-to-qir", "ModuleOp"> {
  let summary = "Convert qssa dialect to Microsoft QIR operations in LLVM dialect";
  let constructor = "mlir::createConvertQuantumToLLVMPass()";
  let dependentDialects = ["quantum::QuantumDialect",
                           "StandardOpsDialect",
                           "AffineDialect",
                           "scf::SCFDialect",
                           "LLVM::LLVMDialect"];
}

#endif // CONVERSION_QUANTUMTOLLVM_CONVERTQUANTUMTOLLVM
