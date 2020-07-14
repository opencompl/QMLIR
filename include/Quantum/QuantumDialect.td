//===- QuantumDialect.td - Quantum dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quantum dialect definition.
//===----------------------------------------------------------------------===//

def Quantum_Dialect : Dialect {
    let name = "quantum";
    let summary = "A quantum out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "quantum";
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Quantum_Dialect, mnemonic, traits>;

#endif // QUANTUM_DIALECT
