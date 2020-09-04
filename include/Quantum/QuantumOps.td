//===- QuantumOps.td - Quantum dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "mlir/IR/OpBase.td"

// include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
// include "mlir/IR/OpAsmInterface.td"
// include "mlir/Interfaces/CallInterfaces.td"
// include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Interfaces/VectorInterfaces.td"
// include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
def Quantum_QubitType :
    Type<CPred<"$_self.isa<QubitType>()">, "Quantum Qubit Type">;

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
// Transform Op
//===----------------------------------------------------------------------===//

def Quantum_TransformOp : Quantum_Op<"transform", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Apply a gate to a qubit array";
    let description = [{
        This operation takes a qubit array of type `qubit<N>` and 
        a gate attribute, and applies the gate to the qubit.
        It returns the same qubit array.

        TODO: specification for gate attribute

        Example:

        ```mlir
        #gate = { name = "X", matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xf32> }
        %q1 = quantum.allocate : !quantum.qubit<4>
        %q2 = quantum.transform #gate(%q1) : !quantum.qubit<4>
        ```
    }];

    let arguments = (ins
        DictionaryAttr:$gate,
        Quantum_QubitType:$qinp);

    let results = (outs 
        Quantum_QubitType:$qout);

    let assemblyFormat = [{
        $gate `(` $qinp `)` attr-dict `:` type($qinp)
    }];

    let verifier = [{
        return success();
    }];
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
        /// TODO: rewrite check (currently works because both dynamic sizes use -1)
        if (qinp().getType().cast<QubitType>().getSize() 
            != res().getType().cast<MemRefType>().getShape()[0])
            return emitOpError("Qubit and result sizes don't match.");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Foo Op (for testing)
//===----------------------------------------------------------------------===//

def Quantum_FooOp : Quantum_Op<"foo", [SameOperandsAndResultType]> {
    // let summary = "Testing Operation";
    // let description = [{
    //     I'm trying to understand MLIR and Tablegen...
    // }];

    let arguments = (ins Quantum_QubitType:$qinp, DictionaryAttr:$val);
    let results = (outs Quantum_QubitType:$qout);

    let assemblyFormat = [{
        $val `(` $qinp `)` attr-dict `:` type($qinp)
    }];

    let verifier = [{
        return success();
    }];
}

#endif // QUANTUM_OPS
