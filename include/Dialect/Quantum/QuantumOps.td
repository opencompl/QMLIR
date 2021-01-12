//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "mlir/IR/OpBase.td"
// include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
// include "mlir/IR/OpAsmInterface.td"
// include "mlir/Interfaces/CallInterfaces.td"
// include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Interfaces/VectorInterfaces.td"
// include "mlir/Interfaces/ViewLikeInterface.td"

include "QuantumDialect.td"
include "QuantumTypes.td"

//===----------------------------------------------------------------------===//
// Allocate Op
//===----------------------------------------------------------------------===//

def Quantum_AllocateOp : Quantum_Op<"allocate", [NoSideEffect]> {
    let summary = "Allocate an array of qubits.";
    let description = [{
        This operation takes no arguments,
        and returns a qubit array `qubit<N>`.

        Example:

        ```mlir
        // allocate a fixed number of qubits
        %q0 = quantum.allocate() : !quantum.qubit<10>

        // allocate a variable number of qubits
        %n = constant 10 : index
        %q1 = quantum.allocate(%n) : !quantum.qubit<?>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicsize);
    let results = (outs Quantum_QubitType:$qout);

    let verifier = [{
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Dimension Op
//===----------------------------------------------------------------------===//

def Quantum_DimensionOp : Quantum_Op<"dim", [NoSideEffect, AllTypesMatch<["qinp", "qout"]>]> {
    let summary = "Get the dimension of a qubit array";
    let description = [{
        This operation takes a qubit array
        and returns the size as an IndexType

        Example:

        ```mlir
        func @use_dim(%qs: !quantum.qubit<?>) : !quantum.qubit<?> {
            %qs1, %n = quantum.dim %qs : !quantum.qubit<?>
            // use %n
            return %qs1 : !quantum.qubit<?> 
        }
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_QubitType:$qout, Index:$size);

    let assemblyFormat = [{
        $qinp attr-dict `:` type($qinp)
    }];

    let verifier = [{
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Cast Op
//===----------------------------------------------------------------------===//

def Quantum_CastOp : Quantum_Op<"cast", [NoSideEffect]> {
    let summary = " an array of qubits.";
    let description = [{
        Convert a qubit array to an equivalent type.
        Takes one argument, and returns the converted qubit array.
        Useful for converting to variable length arrays,
        to pass to library functions etc.

        Example:

        ```mlir
        %0 = quantum.allocate() : !quantum.qubit<10>
        %1 = quantum.cast %0 : !quantum.qubit<10> to !quantum.qubit<?>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_QubitType:$qout);

    let assemblyFormat = [{
        $qinp attr-dict `:` type($qinp) `to` type($qout)
    }];

    let extraClassDeclaration = [{
        /// Return true if `a` and `b` are valid operand and result pairs for
        /// the operation.
        static bool areCastCompatible(Type a, Type b) {
            QubitType qa = a.cast<QubitType>();
            QubitType qb = b.cast<QubitType>();

            if (qa.hasStaticSize() && qb.hasStaticSize() && qa.getSize() != qb.getSize()) {
                return false;
            }
            return true;
        }
    }];

    let verifier = [{
        if (!areCastCompatible(qinp().getType(), qout().getType()))
            return emitOpError("Incompatible types for cast.");

        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Concat Op
//===----------------------------------------------------------------------===//

def Quantum_ConcatOp : Quantum_Op<"concat", [NoSideEffect]> {
    let summary = "Concatenate two arrays of qubits.";
    let description = [{
        This operation takes two qubit arrays and concatenates them into one.

        Example:

        ```mlir
        %q1 = quantum.allocate() : !quantum.qubit<4>
        %q2 = quantum.allocate() : !quantum.qubit<5>
        %n = constant 3 : index
        %q3 = quantum.allocate(%n) : !quantum.qubit<?>

        %qall = quantum.concat %q1, %q2, %q3 : (!quantum.qubit<4>, !quantum.qubit<5>, !quantum.qubit<?>) -> !quantum.qubit<9>
        ```
    }];

    let arguments = (ins Variadic<Quantum_QubitType>:$qinp);
    let results = (outs Quantum_QubitType:$qout);
}

//===----------------------------------------------------------------------===//
// Split Op
//===----------------------------------------------------------------------===//

def Quantum_SplitOp : Quantum_Op<"split", [NoSideEffect]> {
    let summary = "Split an array of qubits into two disjoint arrays";
    let description = [{
        This operation takes a qubit array, and splits it into two arrays.
        For every dynamic result type, it takes a mandatory index type size argument.

        Examples:

        ```mlir
        func @try_split(%q : !quantum.qubit<10>) {
          %q1, %q2 = quantum.split %q : !quantum.qubit<10> -> (!quantum.qubit<5>, !quantum.qubit<5>)
          %1 = constant 1 : index
          %2 = constant 2 : index
          %q3:3 = quantum.split %q1[%1, %2, %2] : !quantum.qubit<5> -> (!quantum.qubit<?>, !quantum.qubit<?>, !quantum.qubit<?>)
          return
        }
        ```
    }];

    let arguments = (ins
        Quantum_QubitType:$qinp,
        Variadic<Index>:$sizes);

    let results = (outs 
        Variadic<Quantum_QubitType>:$qouts);
}

//===----------------------------------------------------------------------===//
// Measure Op
//===----------------------------------------------------------------------===//

def Quantum_MeasurementType : MemRefRankOf<[I1], [1]>;

def Quantum_MeasureOp : Quantum_Op<"measure", [NoSideEffect]> {
    let summary = "Measure an array of qubits.";
    let description = [{
        This operation takes an array of qubits of type `qubit<N>`
        and returns the measurement outcome in the standard basis
        as a 1-D array of bits (`i1`)

        Note: The qubits are consumed and can no longer be used.

        Example:

        ```mlir
        %qs = quantum.allocate : !quantum.qubit<10>
        %res = quantum.measure %qs : !quantum.qubit<10> -> : memref<10xi1>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_MeasurementType:$res);

    let assemblyFormat = [{
        $qinp attr-dict `:` type($qinp) `->` type($res)
    }];

    let verifier = [{
        auto qubitType = qinp().getType().cast<QubitType>();
        auto memRefType = res().getType().cast<MemRefType>();

        if (qubitType.hasStaticSize() != memRefType.hasStaticShape() ||
            (qubitType.hasStaticSize() &&
             qubitType.getSize() != memRefType.getShape()[0]))
            return emitOpError("Qubit and result sizes don't match.");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Primitive Gate Ops
//===----------------------------------------------------------------------===//

// Base Class for primitive gate ops
// If there are no floating point parameters, the `()` are dropped.
//
//   %1 = gate(%a0 : f32, ...) %0 : qubit<?>
//
//
class PrimitiveGateOp<string mnemonic, list<OpTrait> traits = []>
  : Quantum_Op<mnemonic, !listconcat([], traits)> {
  let arguments = (ins
    Variadic<AnyFloat>:$params,
    Quantum_QubitType:$qinp);
  let results = (outs
    Quantum_QubitType:$qout);

  let parser = [{ return ::parsePrimitiveGateOp(parser, result); }];
}

// Base Class for primitive controlled ops
// If there are no floating point parameters, the `()` are dropped.
//
//   %1, %c_out = gate(%a0 : f32, ...) [%c_in : qubit<?>] %0 : qubit<?>
//
class PrimitiveControlledGateOp<string mnemonic, list<OpTrait> traits = []>
  : Quantum_Op<mnemonic, !listconcat([], traits)> {
  let arguments = (ins
    Variadic<AnyFloat>:$params,
    Quantum_QubitType:$qcontrols_inp,
    Quantum_QubitType:$qinp);
  let results = (outs
    Quantum_QubitType:$qcontrols_out,
    Quantum_QubitType:$qout);

  let parser = [{ return ::parsePrimitiveControlledGateOp(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Standard Quantum Gates
//===----------------------------------------------------------------------===//

// Base Class for parameterless gate ops
//
//   %1 = quantum.gate %0 : !quantum.qubit<?>
//
class SimplePrimitiveGateOp<string mnemonic, list<OpTrait> traits = []>
  : PrimitiveGateOp<mnemonic, !listconcat([], traits)> {
}

def PauliXGateOp : SimplePrimitiveGateOp<"pauliX"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli X gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliX %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliX %q0 : !quantum.qubit<?>
      ```
  }];
}

def PauliYGateOp : SimplePrimitiveGateOp<"pauliY"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli Y gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliY %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliY %q0 : !quantum.qubit<?>
      ```
  }];
}

def PauliZGateOp : SimplePrimitiveGateOp<"pauliZ"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli Z gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliZ %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliZ %q0 : !quantum.qubit<?>
      ```
  }];
}

def HadamardGateOp : SimplePrimitiveGateOp<"H"> {
  let summary = "Quantum Hadamard gate";
  let description = [{
      Takes a qubit array, and applies the Hadamard gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.H %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.H %q0 : !quantum.qubit<?>
      ```
  }];
}

def CNOTGateOp : SimplePrimitiveGateOp<"CNOT"> {
  let summary = "Quantum CNOT gate";
  let description = [{
      Takes a 2-qubit array, and applies the CNOT gate,
      with the first qubit as control, and second as target.

      Example:
      ```mlir
      %q1 = quantum.CNOT %q0 : !quantum.qubit<2>
      ```
  }];

  let verifier = [{
    auto type = qinp().getType().cast<QubitType>();
    if (!type.hasStaticSize() || type.getSize() != 2)
      return emitOpError("expected a two qubit array (qubit<2>), got ") << type;
    return ::verify(*this);
  }];
}

//===----------------------------------------------------------------------===//
// Unregistered Quantum Gates
//===----------------------------------------------------------------------===//

// TODO Support for generic gate ops, which can be modified and replaced
//      by IR tranformations.

//===----------------------------------------------------------------------===//
// QASM Dialect Ops
//===----------------------------------------------------------------------===//
#endif // QUANTUM_OPS