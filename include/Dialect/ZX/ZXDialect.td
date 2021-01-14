#ifndef ZX_DIALECT
#define ZX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ZX dialect definition.
//===----------------------------------------------------------------------===//

def ZX_Dialect : Dialect {
  let name = "zx";
  let summary = "Dialect for ZX-calculus description of quantum programs";
  let description = [{
    https://en.wikipedia.org/wiki/ZX-calculus
  }];

  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser & parser) const override;
    void printType(Type type, DialectAsmPrinter & os) const override;
  }];

  let cppNamespace = "::mlir::ZX";
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class ZX_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<ZX_Dialect, mnemonic, traits> {

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

#endif // ZX_DIALECT
