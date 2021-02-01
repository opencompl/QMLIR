//===- QuantumTypes.td - Quantum dialect Types -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZXGRAPH_TYPES
#define ZXGRAPH_TYPES

include "mlir/IR/OpBase.td"

// !zxg.node
def ZXGraph_NodeType
  : Type<CPred<"$_self.isa<NodeType>()">, "ZX Node Type">
  , BuildableType<"NodeType::get($_builder.getContext())">;

#endif // ZXGRAPH_TYPES
