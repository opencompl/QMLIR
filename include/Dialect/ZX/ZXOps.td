//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZX_OPS
#define ZX_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect/ZX/ZXDialect.td"
include "Dialect/ZX/ZXTypes.td"

// `op` [ param : f32 ] ( args... ) : output-type
class ZX_ParametrizedNodeOp<string mnemonic, list<OpTrait> traits = []>
  : ZX_Op<mnemonic, traits> {

  let arguments = (ins AnyFloat:$angle, Variadic<ZX_WireType>:$input);
  let results = (outs Variadic<ZX_WireType>:$output);

  let assemblyFormat = [{
    `[` $angle `:` type($angle) `]` `(` $input `)` attr-dict  `:` type($output)
  }];

  let verifier = [{
    return success();
  }];
}

//====================================
// Specific node types
//====================================

def ZX_ZOp : ZX_ParametrizedNodeOp<"Z"> {
  let summary = "ZX Z node";
  let description = [{
    // %theta: f32
    %d, %e = zx.Z[%theta: f32] (%a, %b, %c)
  }];
}

def ZX_XOp : ZX_ParametrizedNodeOp<"X"> {
  let summary = "ZX X node";
  let description = [{
    // %theta: f32
    %d, %e = zx.X[%theta: f32] (%a, %b, %c)
  }];
}

def ZX_HOp : ZX_Op<"H"> {
  let summary = "ZX Hadamard node";
  let description = [{
    %b = zx.H (%a)
  }];

  let arguments = (ins ZX_WireType:$input);
  let results = (outs ZX_WireType:$output);

  let assemblyFormat = [{
    `(` $input `)` attr-dict
  }];

  let verifier = [{
    return success();
  }];
}

//======================
// Terminating wires
//======================

def ZX_InputNodeOp : ZX_Op<"input"> {
  let summary = "ZX Input wire Op";
  let description = [{
    Create an input wire
    %a = zx.input
  }];

  let results = (outs ZX_WireType:$output);

  let assemblyFormat = [{
    attr-dict
  }];

  let verifier = [{
    return success();
  }];
}

def ZX_OutputNodeOp : ZX_Op<"output"> {
  let summary = "ZX Output wire Op";
  let description = [{
    Sink an existing wire.
    zx.output %a
  }];

  let arguments = (outs ZX_WireType:$input);

  let assemblyFormat = [{
    $input attr-dict
  }];

  let verifier = [{
    return success();
  }];
}

#endif // ZX_OPS
