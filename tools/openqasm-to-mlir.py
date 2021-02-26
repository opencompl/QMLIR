#! env python3

# OpenQASM Lang Spec: https://arxiv.org/pdf/1707.03429v2.pdf

import sys
import logging
import argparse
import copy
from typing import Union

from qiskit.qasm import Qasm
import qiskit.qasm.node as Node

class ConversionError(Exception):
    pass

class UnimplementedError(Exception):
    pass

################################################################################
""" Base class for MLIR Objects """
class MLIRBase:
    def __init__(self):
        pass
    def __str__(self):
        return '\n'.join(self.serialize())
    def indent(self, lines: list[str], size:int=1) -> list[str]:
        return list(map(lambda l: ' ' * (2 * size) + l, lines))
    def build(self, *args, **kwargs):
        pass
    def serialize(self) -> list[str]:
        raise UnimplementedError("Serialize")
    def show(self) -> str:
        raise UnimplementedError("Abstract Method")

############################################################
""" Base class for MLIR Types
Form: [ `!` $dialect `.` ] $name
"""
class MLIRType(MLIRBase):
    name = None
    dialect = 'std'
    def __init__(self, *args, **kwargs):
        self.build(args, kwargs)
    def show(self) -> str:
        prefix = f'!{self.dialect}.' if self.dialect != 'std' else ''
        return prefix + self.name

    def serialize(self) -> list[str]:
        return [self.show()]

""" Specific Type constructors """
class FloatType(MLIRType):
    def build(self, prec: int = 32):
        name = f'f{prec}'
class IntType(MLIRType):
    def build(self, prec: int = 32):
        name = f'i{prec}'
class QubitType(MLIRType):
    name = 'qubit'
    dialect = 'qasm'

class MLIRAttribute(MLIRBase):
    """ Base class for Attributes
    """
    def __init__(self, *args, **kwargs):
        self.dialect: str = kwargs.get('dialect', 'std')
        self.build(*args, **kwargs)
    def serialize(self) -> list[str]:
        return [self.show()]

class FloatAttr(MLIRAttribute):
    def build(self, val: float):
        self.value = val
    def show(self) -> str:
        return str(self.value)
class IntAttr(MLIRAttribute):
    def build(self, val: int):
        self.value = val
    def show(self) -> str:
        return str(self.value)

############################################################
"""SSA Value
Stores the identifier
"""
class SSAValue(MLIRBase):
    def __init__(self, name: str, ty: MLIRType):
        self.name = name
        self.ty = ty
    def show(self, withType: bool=False) -> str:
        if withType:
            return f'%{self.name}: {self.getType()}'
        return f'%{self.name}'
    def serialize(self) -> list[str]:
        return [self.show()]
    def getType(self) -> MLIRType:
        return self.ty

class SSAValueMap:
    def __init__(self):
        self.vmap = dict()
        self.index = 0

    def lookup(self, name: str) -> SSAValue:
        return self.vmap[name]
    def insert(self, name: str, ty: MLIRType) -> SSAValue:
        self.vmap[name] = SSAValue(str(self.index), ty)
        self.index += 1
        return self.vmap[name]
    def insertArray(self, name: str, size: int, ty: MLIRType):
        for i in range(size):
            self.insert(f'{name}:{i}', ty)

############################################################
class MLIROperation(MLIRBase):
    """Base class for all Operations

    Each operation inherits from this.
    And must define the following:
        - name : str
        - show() -> str
            Prints the operation. Does not include operation name
            Example: for AddFOp - '%1, %2 : f32'

    And optionally the following:
        - dialect : str
        - build(*args) -> ()
        - serialize() -> list[str]
    """
    name = None
    dialect = 'std'
    operands: list[SSAValue] = []
    attributes: list[MLIRAttribute] = []
    results: list[SSAValue] = []
    def __init__(self, valueMap: SSAValueMap, *args, **kwargs):
        self.valueMap = valueMap
        self.build(*args, **kwargs)
        pass

    def serialize(self) -> list[str]:
        return [f'{self.dialect}.{self.name} {self.show()}']
    def build(self, *args, **kwargs):
        pass

    def show(self) -> str:
        raise UnimplementedError("Abstract Method")

""" std Ops """
class ReturnOp(MLIROperation):
    name = 'return'
    def show(self) -> str:
        return ''

class ConstantOp(MLIROperation):
    name = 'constant'
    def build(self, val: Union[float, int]):
        if isinstance(val, float):
            self.attributes.append(FloatAttr(val))
        if isinstance(val, int):
            self.attributes.append(IntAttr(val))
    def getValue(self) -> MLIRAttribute:
        return self.attributes[0]
    def getType(self) -> MLIRType:
        return self.results[0].getType()
    def show(self) -> str:
        return f'{self.getValue()} : {self.getType()}'

class UnaryOp(MLIROperation):
    def build(self, arg: SSAValue):
        pass

class SinOp(UnaryOp):
    dialect = 'math'
    name = 'sin'
class CosOp(UnaryOp):
    dialect = 'math'
    name = 'cos'
class TanOp(UnaryOp):
    dialect = 'math'
    name = 'tan'
class ExpOp(UnaryOp):
    dialect = 'math'
    name = 'exp'
class LnOp(UnaryOp):
    dialect = 'math'
    name = 'log'
class SqrtOp(UnaryOp):
    dialect = 'math'
    name = 'sqrt'

class BinaryOp(MLIROperation):
    dialect = 'std'
    name = ''
    def __init__(self, lhs: SSAValue, rhs: SSAValue, res: SSAValue):
        super().__init__(f'{self.dialect}.{self.name}'
                f'{lhs.show()}, {rhs.show()} : {res.ty.show()}')
        self.lhs = lhs
        self.rhs = rhs
        self.res = res

class AddFOp(BinaryOp):
    name = 'addf'
class SubFOp(BinaryOp):
    name = 'subf'
class MulFOp(BinaryOp):
    name = 'mulf'

""" qasm Ops """
class UOp(MLIROperation):
    def __init__(self, theta: SSAValue, phi: SSAValue,
                       lambd: SSAValue, qubit: SSAValue):
        super().__init__(f'qasm.U ({theta.show(True)}, {phi.show(True)},'
                f'{lambd.show(True)}) {qubit}')
        self.theta = theta
        self.phi = phi
        self.lambd = lambd
        self.qubit = qubit

############################################################
ExpressionType = Union[Node.Id, Node.BinaryOp]
OperationType = Union[Node.UniversalUnitary, Node.CustomUnitary]

class MLIRBlock(MLIRBase):
    def __init__(self, valueMap: SSAValueMap = SSAValueMap()):
        self.valueMap: SSAValueMap = copy.deepcopy(valueMap)
        self.body: list[MLIROperation] = []

    def serialize(self) -> list[str]:
        code = []
        for op in self.body:
            code += op.serialize()
        return code

    def addOp(self, op: MLIROperation):
        self.body.append(op)
    def buildOp(self, opClass: type, *args) -> list[SSAValue]:
        op = opClass(self.valueMap, *args)
        self.addOp(op)
        return op.getResults()

    def parseExpression(self, op: ExpressionType) -> SSAValue:
        if isinstance(op, Node.BinaryOp):
            binop = op.children[0]
            lhs = self.parseExpression(op.children[1])
            rhs = self.parseExpression(op.children[2])
            print(binop)
            print(binop.name)
            sys.exit(0)
        if isinstance(op, Node.Id):
            return self.valueMap.lookup(op.name)
        if isinstance(op, Node.Real):
            constOp = ConstantOp(op.value, FloatType())
            self.addOp(constOp)
            return 
        raise UnimplementedError(f"Unknown expression kind {type(op)}: {op.qasm()}")

    def parseOperation(self, op: OperationType):
        if isinstance(op, Node.UniversalUnitary):
            args = op.children[0].children
            theta = self.parseExpression(args[0])
            phi   = self.parseExpression(args[1])
            lambd = self.parseExpression(args[2])
            qubit = self.valueMap.insert(op.children[1].name, QubitType())
            self.body.append(UOp(theta, phi, lambd, qubit))
            return qubit
        if isinstance(op, Node.CustomUnitary):
            pass
        raise UnimplementedError(f"Unknown operation kind {type(op)}: {op.qasm()}")

class MLIRFunction(MLIRBase):
    def __init__(self,
            name: str,
            arguments: list[SSAValue],
            results: list[MLIRType],
            body: MLIRBlock):
        self.name = name
        self.arguments = arguments
        self.results = results
        self.body = body

    def serialize(self) -> list[str]:
        args = ', '.join(map(lambda v: v.show(True), self.arguments))
        results = ', '.join(map(lambda t: t.show(), self.arguments))
        code = [f'func @{self.name}({args}) -> ({results}) {{']
        code += self.body.serialize()
        code.append('}')
        return code

class MLIRModule(MLIRBase):
    def __init__(self):
        self.declarations: list[MLIRFunction] = []

    def serialize(self) -> list[str]:
        code = ['module {']
        for decl in self.declarations:
            code += self.indent(decl.serialize())
        code.append('}')
        return code

    def addDecl(self, decl: MLIRFunction):
        self.declarations.append(decl)

################################################################################


def QASMToMLIR(code: str, outputFile, strict=False):
    try:
        src = Qasm(data=code).parse()
    except:
        raise ConversionError("Could not load input file")

    module: MLIRModule = MLIRModule()
    mainBlock = MLIRBlock()
    for node in src.children:
        if isinstance(node, Node.Format):
            # version string
            if strict:
                raise UnimplementedError("Version String")
        elif isinstance(node, Node.Gate):
            # Gate definition
            name: str = node.name
            args: list[SSAValue] = []
            results: list[MLIRType] = []
            body: MLIRBlock = MLIRBlock()
            localScope: SSAValueMap = body.valueMap

            for arg in node.arguments.children:
                val = localScope.insert(arg.name, FloatType())
                args.append(val)
            for arg in node.bitlist.children:
                val = localScope.insert(arg.name, QubitType())
                args.append(val)
            for op in node.body.children:
                body.parseOperation(op)
            body.addOp(ReturnOp())

            gate: MLIRFunction = MLIRFunction(name, args, results, body)
            module.addDecl(gate)
        elif isinstance(node, OperationType):
            mainBlock.parseOperation(node)
        else:
            raise ConversionError(f"Unknown node object found at line {node.line}: {type(node)}")


def main():
    parser = argparse.ArgumentParser(description='QASM to MLIR translation tool')
    parser.add_argument('-i', metavar='input', dest='input', type=str,
            help='Input file (uses stdin if not specified)', required=False)
    parser.add_argument('-o', metavar='output', dest='output', type=str,
            help='Output file (uses stdout if not specified)', required=False)
    parser.add_argument('-v', action='store_true', dest='verbose',
            help='verbose', required=False)
    args = parser.parse_args()

    if args.input is None: args.input = sys.stdin
    else: args.input = open(args.input, 'r')
    if args.output is None: args.output = sys.stdout
    else: args.output = open(args.output, 'w+')

    code = args.input.read()
    QASMToMLIR(code, args.output, strict=args.verbose)

if __name__ == "__main__":
    main()
