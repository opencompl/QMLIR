#ifndef ZXGRAPH_DIALECT
#define ZXGRAPH_DIALECT
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ZX Graph dialect definition.
//===----------------------------------------------------------------------===//

def ZXGraph_Dialect : Dialect {
  let name = "zxg";
  let summary = "Dialect for ZX-calculus description of quantum programs";
  let description = [{
    Undirected graph version of the ZX dialect
    First declare nodes, and then join them using wires
  }];

  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser & parser) const override;
    void printType(Type type, DialectAsmPrinter & os) const override;
  }];

  let cppNamespace = "::mlir::ZXGraph";
}

//===----------------------------------------------------------------------===//
// Base ZX operation definition.
//===----------------------------------------------------------------------===//

class ZXGraph_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<ZXGraph_Dialect, mnemonic, traits> {

  // For each op, the following static functions need to be defined
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
  //                                               OperationState &state);
  // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
  // * static LogicalResult verify(<op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
}

#endif // ZXGRAPH_DIALECT
