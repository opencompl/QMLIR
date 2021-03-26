#ifndef QASM_DIALECT
#define QASM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// QASM dialect definition.
//===----------------------------------------------------------------------===//

def QASM_Dialect : Dialect {
  let name = "qasm";
  let summary = "Dialect for OpenQASM";
  let description = [{
    Dialect to represent OpenQASM programs.
      Spec Version 3.0: https://qiskit.github.io/openqasm/
      - Version 2.0: https://github.com/Qiskit/openqasm/tree/OpenQASM2.x
  }];

  let extraClassDeclaration = [{
    static StringRef getGateAttrName() { return "qasm.gate"; };
    static StringRef getQelibGateAttrName() { return "qasm.stdgate"; };
    static StringRef getMainFuncAttrName() { return "qasm.main"; };
  }];

  let cppNamespace = "::mlir::QASM";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class QASM_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<QASM_Dialect, mnemonic, traits> {

    // For each QASM op, the following static functions need to be defined in
    // QASMOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

class QASM_Type<string name> : TypeDef<QASM_Dialect, name> {}

#endif // QASM_DIALECT
