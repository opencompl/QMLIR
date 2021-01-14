//===- QuantumTypes.td - Quantum dialect Types -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZX_TYPES
#define ZX_TYPES

include "mlir/IR/OpBase.td"

// !zx.wire
def ZX_WireType
  : Type<CPred<"$_self.isa<WireType>()">, "ZX Wire Type">
  , BuildableType<"WireType::get($_builder.getContext())">;

#endif // ZX_TYPES
