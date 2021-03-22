//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZXGRAPH_OPS
#define ZXGRAPH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect/ZXGraph/ZXGraphDialect.td"
include "Dialect/ZXGraph/ZXGraphTypes.td"

//===========================================================================//
// ZXGraph Dialect Ops
//===========================================================================//

def ZXGraph_TerminalNodeOp : ZXGraph_Op<"terminal"> {
  let summary = "ZXGraph Terminal node Op";
  let results = (outs ZXGraph_NodeType:$output);
  let assemblyFormat = [{ attr-dict }];
  let verifier = [{ return success(); }];
}

class ZXGraph_ParametrizedNodeOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]>
  : ZXGraph_Op<mnemonic, traits> {
  let arguments = (ins AnyFloat:$param);
  let results = (outs ZXGraph_NodeType:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$val), [{
      build($_builder, $_state, NodeType::get($_builder.getContext()), val);
    }]>
  ];
  let assemblyFormat = [{ $param attr-dict `:` type($param) }];
  let verifier = [{ return success(); }];
}

def ZXGraph_ZNodeOp : ZXGraph_ParametrizedNodeOp<"Z"> {
  let summary = "ZXGraph Z node Op";
}
def ZXGraph_XNodeOp : ZXGraph_ParametrizedNodeOp<"X"> {
  let summary = "ZXGraph X node Op";
}

def ZXGraph_HNodeOp : ZXGraph_Op<"H", [NoSideEffect]> {
  let summary = "ZXGraph Hadamard node Op";
  let results = (outs ZXGraph_NodeType:$output);
  let assemblyFormat = [{ attr-dict }];
  let verifier = [{ return success(); }];
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, NodeType::get($_builder.getContext()));
    }]>
  ];
}

def ZXGraph_WireOp : ZXGraph_Op<"wire"> {
  let summary = "ZXGraph Wire Op";
  let arguments = (ins ZXGraph_NodeType:$lhs, ZXGraph_NodeType:$rhs);
  let assemblyFormat = [{ $lhs $rhs attr-dict }];
  let verifier = [{ return success(); }];
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    ::mlir::Value getOtherOperand(::mlir::Value operand) {
      if (operand == lhs())
        return rhs();
      return lhs();
    }
  }];
}

#endif // ZXGRAPH_OPS
