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

//===========================================================================//
// ZX Dialect Ops
//===========================================================================//

// `op` [ param : f32 ] ( args... ) : output-type
class ZX_GenericNodeOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]>
  : ZX_Op<mnemonic, traits> {
  let description = [{
    Inputs:
      - floating parameter
      - Any number of wires
    Outputs:
      - Any number of wires
  }];

  let arguments = (ins AnyFloat:$param, Variadic<ZX_WireType>:$inputWires);
  let results = (outs Variadic<ZX_WireType>:$outputWires);

  let assemblyFormat = [{
    `(` $param `:` type($param) `)` $inputWires attr-dict `:` type($outputWires)
  }];
  let verifier = ?;
}

// Specific node types
def ZX_ZOp : ZX_GenericNodeOp<"Z"> {
  let summary = "ZX Z node";
}
def ZX_XOp : ZX_GenericNodeOp<"X"> {
  let summary = "ZX X node";
}
def ZX_HOp : ZX_Op<"H", [NoSideEffect]> {
  let summary = "ZX Hadamard node";
  let arguments = (ins ZX_WireType:$inputWire);
  let results = (outs ZX_WireType:$outputWire);
  let assemblyFormat = [{ $inputWire attr-dict }];
  let verifier = ?;
}
def ZX_SourceNodeOp : ZX_Op<"source", []> {
  let summary = "ZX Input/Source wire Op";
  let results = (outs ZX_WireType:$outputWire);
  let assemblyFormat = [{ attr-dict }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state,
                   ::mlir::ZX::WireType::get($_builder.getContext()));
    }]>
  ];
}

def ZX_SinkNodeOp : ZX_Op<"sink", []> {
  let summary = "ZX Output/Sink wire Op";
  let arguments = (ins ZX_WireType:$inputWire);
  let assemblyFormat = [{ $inputWire attr-dict }];
  let verifier = ?;
}
def ZX_MeasureOp : ZX_Op<"measure", [NoSideEffect]> {
  let summary = "ZX measure Op";
  let arguments = (ins ZX_WireType:$inputWire);
  let results = (outs I1:$mres, ZX_WireType:$outputWire);
  let assemblyFormat = [{ $inputWire attr-dict }];
  let verifier = ?;
}

#endif // ZX_OPS
