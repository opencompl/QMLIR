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
      - Optional floating parameter (angle, as a fraction of pi)
      - Any number of wires
    Outputs:
      - Any number of wires

    Getters:
      - hasParam() -> bool
      - getParam() -> Value
      - getInputWires() -> Values
      - getResults() -> Values
  }];

  let arguments = (ins Variadic<AnyType>:$input);
  let results = (outs Variadic<ZX_WireType>:$output);

  let assemblyFormat = [{
    `(` $input  `)` attr-dict `:` functional-type($input, results)
  }];
  let verifier = [{
    return success();
  }];

  let extraClassDeclaration = [{
    bool hasParam() {
      auto operands = getODSOperands(0);
      if (operands.empty()) return false;
      auto param = *operands.begin();
      return param.getType().isa<FloatType>();
    }
    ::mlir::Value getParam() {
      assert(hasParam() && "No floating point parameter provided!");
      return *getODSOperands(0).begin();
    }
    ::mlir::Operation::operand_range getInputWires() {
      unsigned st = hasParam() ? 1 : 0;
      return {std::next(getOperation()->operand_begin(), st),
              getOperation()->operand_end()};
    }
  }];
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
  let arguments = (ins ZX_WireType:$input);
  let results = (outs ZX_WireType:$output);
  let assemblyFormat = [{ $input attr-dict }];
  let verifier = [{ return success(); }];
}
def ZX_SourceNodeOp : ZX_Op<"source", []> {
  let summary = "ZX Input/Source wire Op";
  let results = (outs ZX_WireType:$output);
  let assemblyFormat = [{ attr-dict }];
  let verifier = [{ return success(); }];
}

def ZX_SinkNodeOp : ZX_Op<"sink", []> {
  let summary = "ZX Output/Sink wire Op";
  let arguments = (ins ZX_WireType:$input);
  let assemblyFormat = [{ $input attr-dict }];
  let verifier = [{ return success(); }];
}

#endif // ZX_OPS
