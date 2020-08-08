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

// include "mlir/IR/OpAsmInterface.td"
// include "mlir/Interfaces/CallInterfaces.td"
// include "mlir/Interfaces/ControlFlowInterfaces.td"
// include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Interfaces/VectorInterfaces.td"
// include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
def Quantum_QubitType :
    Type<CPred<"$_self.isa<QubitType>()">, "Quantum Qubit Type">;

def Quantum_GateType :
    Type<CPred<"$_self.isa<GateType>()">, "Quantum Gate Type">;

//===----------------------------------------------------------------------===//
// Allocate Op
//===----------------------------------------------------------------------===//

def Quantum_AllocateOp : Quantum_Op<"allocate", []> {
    let summary = "Allocate an array of qubits.";
    let description = [{
        This operation takes no arguments,
        and returns a qubit array `qubit<N>`.

        Example:

        ```mlir
        // allocate 10 qubits
        %qs = quantum.allocate : !quantum.qubit<10>
        ```
    }];

    let results = (outs Quantum_QubitType:$qout);

    let assemblyFormat = [{
        attr-dict `:` type($qout)
    }];

    let verifier = [{
        if (!qout().getType().cast<QubitType>().hasStaticSize()) {
            return emitOpError("Can only allocate fixed size qubit arrays");
        }
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Cast Op
//===----------------------------------------------------------------------===//

def Quantum_CastOp : Quantum_Op<"cast", []> {
    let summary = " an array of qubits.";
    let description = [{
        Convert a qubit array to an equivalent type.
        Takes one argument, and returns the converted qubit array.
        Useful for converting to variable length arrays,
        to pass to library functions etc.

        Example:

        ```mlir
        %0 = quantum.allocate : !quantum.qubit<10>
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

def Quantum_ConcatOp : Quantum_Op<"concat", []> {
    let summary = "Concatenate two arrays of qubits.";
    let description = [{
        This operation takes two qubit arrays of types `qubit<N>` and 
        `qubit<M>`, and returns the concatenated array of type `qubit<N+M>`

        Example:

        ```mlir
        %q1 = quantum.allocate : !quantum.qubit<4>
        %q2 = quantum.allocate : !quantum.qubit<5>
        %q3 = quantum.concat %q1, %q2 : (!quantum.qubit<4>, !quantum.qubit<5>) -> !quantum.qubit<9>
        ```
    }];

    let arguments = (ins
        Quantum_QubitType:$qinp0, 
        Quantum_QubitType:$qinp1);

    let results = (outs Quantum_QubitType:$qout);

    let assemblyFormat = [{
        $qinp0 `,` $qinp1 attr-dict `:` `(` type($qinp0) `,` type($qinp1) `)` `->` type($qout)
    }];

    let verifier = [{
        // if (qinp0().getType().cast<QubitType>().getSize()
        //     + qinp1().getType().cast<QubitType>().getSize()
        //     != qout().getType().cast<QubitType>().getSize())
        //     return emitOpError("sizes don't match!");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Split Op
//===----------------------------------------------------------------------===//

def Quantum_SplitOp : Quantum_Op<"split", []> {
    let summary = "Split an array of qubits into two";
    let description = [{
        This operation takes a qubit array of type `qubit<N>` and
        returns two arrays `qubit<M>` and `qubit<R>` where `N = M + R` 

        Example:

        ```mlir
        %q1 = quantum.allocate : !quantum.qubit<10>
        %q2 = quantum.split %q1 : !quantum.qubit<10> -> (!quantum.qubit<6>, !quantum.qubit<4>)
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);

    let results = (outs 
        Quantum_QubitType:$qout0, 
        Quantum_QubitType:$qout1);

    let assemblyFormat = [{
        $qinp attr-dict `:` type($qinp) `->` `(` type($qout0) `,` type($qout1) `)` 
    }];

    let verifier = [{
        // if (qout0().getType().cast<QubitType>().getSize()
        //     + qout1().getType().cast<QubitType>().getSize()
        //     != qinp().getType().cast<QubitType>().getSize())
        //     return emitOpError("sizes don't match!");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Transform Op
//===----------------------------------------------------------------------===//

def Quantum_TransformOp : Quantum_Op<"transform", [SameOperandsAndResultType]> {
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
}

//===----------------------------------------------------------------------===//
// Measure Op
//===----------------------------------------------------------------------===//

def Quantum_MeasurementType : 1DTensorOf<[I1]>;
// VectorOfRankAndType<[1], [I1]>;

def Quantum_MeasureOp : Quantum_Op<"measure", []> {
    let summary = "Measure an array of qubits.";
    let description = [{
        This operation takes an array of qubits of type `qubit<N>`
        and returns the measurement outcome in the standard basis
        as a 1-D tensor of bits (`i1`)

        Note: The qubits are consumed and can no longer be used.

        Example:

        ```mlir
        %qs = quantum.allocate : !quantum.qubit<10>
        %res = quantum.measure %qs : !quantum.qubit<10> -> : tensor<10xi1>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_MeasurementType:$res);

    let assemblyFormat = [{
        $qinp attr-dict `:` type($qinp) `->` type($res)
    }];

    let verifier = [{
        if (qinp().getType().cast<QubitType>().getSize() 
            != res().getType().cast<RankedTensorType>().getShape()[0])
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
}

#endif // QUANTUM_OPS
