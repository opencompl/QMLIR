//===- QuantumOps.h - Quantum dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZXGRAPH_ZXOPS_H
#define ZXGRAPH_ZXOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ZXGraphTypes.h"

#define GET_OP_CLASSES
#include "Dialect/ZXGraph/ZXGraphOps.h.inc"

#endif // ZXGRAPH_ZXOPS_H
