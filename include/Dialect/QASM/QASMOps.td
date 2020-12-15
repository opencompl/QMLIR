//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QASM_OPS
#define QASM_OPS

include "mlir/IR/OpBase.td"
// include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
// include "mlir/IR/OpAsmInterface.td"
// include "mlir/Interfaces/CallInterfaces.td"
// include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Interfaces/VectorInterfaces.td"
// include "mlir/Interfaces/ViewLikeInterface.td"

include "QASMDialect.td"
include "QASMTypes.td"

#endif // QASM_OPS
