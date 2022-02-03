# OpenQASM Lang Spec: https://arxiv.org/pdf/1707.03429v2.pdf

from __future__ import annotations
import sys
import logging
import argparse
import copy
from typing import Union

from qiskit.qasm import Qasm
import qiskit.qasm.node as Node

def setupLogger(lev):
    logger = logging.getLogger('Notebook')

    if 'loggerSetupDone' in globals(): return logger

    global loggerSetupDone
    loggerSetupDone = True

    logger.setLevel(lev)

    # https://docs.python.org/3/howto/logging.html#configuring-logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

logger = setupLogger(logging.WARNING)

class ConversionError(Exception):
    pass

class UnimplementedError(Exception):
    pass

def showtree(node, indent=0):
    if type(node) is str:
        node = Qasm(data=node).parse()
    pref = ' ' * (indent*4)
    logger.debug(f'{pref}{type(node)}')
    for child in node.children:
        showtree(child, indent + 1)

""" Base class for MLIR Objects """
class MLIRBase:
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return '\n'.join(self.serialize())
    def indent(self, lines: list[str], size:int=1) -> list[str]:
        return list(map(lambda l: ' ' * (2 * size) + l, lines))
    def build(self):
        pass
    def serialize(self) -> list[str]:
        raise UnimplementedError("Serialize")
    def show(self) -> str:
        raise UnimplementedError("Abstract Method")

""" Base class for MLIR Types
Form: [ `!` $dialect `.` ] $name
"""
class MLIRType(MLIRBase):
    name: str = '<unknown_op>'
    dialect: str = 'std'
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.build()
    def show(self) -> str:
        prefix = f'!{self.dialect}.' if self.dialect != 'std' else ''
        return prefix + self.name

    def serialize(self) -> list[str]:
        return [self.show()]

""" Specific Type constructors """
class FloatType(MLIRType):
    """build(<prec>)
    """
    def build(self):
        prec: int = 64
        if len(self.args) >= 1:
            prec = self.args[0]
        self.name = f'f{prec}'
class IntType(MLIRType):
    """build(<prec>)
    """
    def build(self):
        prec: int = 32
        if len(self.args) >= 1:
            prec = self.args[0]
        self.name = f'i{prec}'

class IndexType(MLIRType):
    name = 'index'

class MemrefType(MLIRType):
    """build(dims: list[Union[None, int]], ty: MLIRType)"""
    name = 'memref'
    def getElementType(self) -> MLIRType:
        return self.ty
    def getSize(self, dim: int) -> int:
        return self.dims[dim]
    def build(self):
        self.dims = self.kwargs['dims']
        self.ty = self.kwargs['ty']
    def show(self) -> str:
        dims = 'x'.join(map(lambda n: '?' if n is None else str(n), self.dims))
        return f'{self.name}<{dims}x{self.ty}>'

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
        self.value = float(val)
    def show(self) -> str:
        return '{:e}'.format(self.value)
class IntAttr(MLIRAttribute):
    def build(self, val: int):
        self.value = val
    def show(self) -> str:
        return str(self.value)
class UnitAttr(MLIRAttribute):
    def build(self):
        pass
    def show(self) -> str:
        return ''
class StringAttr(MLIRAttribute):
    def build(self, val: str):
        self.value = val
    def show(self) -> str:
        return f'"{self.value}"'


"""SSA Value
Stores the identifier
"""
class SSAValue(MLIRBase):
    def __init__(self, name: str, ty: MLIRType):
        self.name = name
        self.ty = ty
    def show(self, withType: bool=False) -> str:
        if withType:
            return f'%{self.name} : {self.getType()}'
        return f'%{self.name}'
    def serialize(self) -> list[str]:
        return [self.show()]
    def getType(self) -> MLIRType:
        return self.ty

class SSAValueMap:
    def __init__(self):
        self.vmap = dict() # values
        self.amap = dict() # arrays
        self.index = 0

    def newValue(self, ty: MLIRType, name: str = '') -> SSAValue:
        """Create a new SSA Value of the given type, and optional name
        """
        val = SSAValue(str(self.index), ty)
        if name:
            self.vmap[name] = val
        self.index += 1
        return val
    def newArray(self, name: str, size: int, ty: MLIRType):
        """Create a new array of SSA values of the given type and name
        """
        self.amap[name] = []
        for i in range(size):
            val = self.newValue(ty, f'{name}_{i}')
            self.amap[name].append(val)
    def insert(self, name: str, value: SSAValue):
        """Add an existing SSA Value to the map
        """
        self.vmap[name] = value
    def insertArray(self, name: str, values: list[SSAValue]):
        """Add a list of existing SSA Values to the map
        """
        size = len(values)
        for i in range(size):
            self.insert(f'{name}_{i}', values[i])
        self.amap[name] = copy.copy(values)

    def lookup(self, name: str) -> SSAValue:
        return self.vmap[name]
    def lookupArray(self, name: str) -> list[SSAValue]:
        return self.amap[name]

    def resolve(self, name: str) -> list[SSAValue]:
        """Resolve a variable name
        Lookup through both the value and array maps
        """
        if name in self.vmap:
            return [self.vmap[name]]
        if name in self.amap:
            return self.amap[name]
        raise ConversionError("Invalid SSA Value")


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
    def __init__(self, valueMap: SSAValueMap, *args, **kwargs):
        self.valueMap = valueMap
        self.args = args
        self.kwargs = kwargs
        self.operands: list[SSAValue] = []
        self.attributes: list[MLIRAttribute] = []
        self.results: list[SSAValue] = []
        self.blocks = []
        self.build()

    def serialize(self) -> list[str]:
        res = ', '.join(map(lambda v: v.show(), self.results))
        op = f'{self.dialect}.{self.name} {self.show()}'
        if len(res) == 0:
            return [op]
        return [res + ' = ' + op]
    def build(self):
        pass

    def show(self) -> str:
        raise UnimplementedError("Abstract Method")

    def addResult(self, ty: MLIRType) -> SSAValue:
        res = self.valueMap.newValue(ty)
        self.results.append(res)
        return res
    def addAttribute(self, attr: MLIRAttribute):
        self.attributes.append(attr)
    def addOperand(self, arg: SSAValue):
        self.operands.append(arg)
    def addBlock(self, block):
        self.blocks.append(block)

    def getResults(self) -> list[SSAValue]:
        return self.results

""" std Ops """
class ReturnOp(MLIROperation):
    name = 'return'
    def build(self):
        self.isGate = ('isGate' in self.kwargs)
    def show(self) -> str:
        return ('{qasm.gate_end}' if self.isGate else '')

class CallOp(MLIROperation):
    """build(func: str, operands: list[SSAValue], results: list[MLIRType])
    Use only named arguments.
    """
    name = 'call'
    def build(self):
        if 'operands' in self.kwargs:
            for arg in self.kwargs['operands']:
                assert(isinstance(arg, SSAValue))
                self.addOperand(arg)
        if 'results' in self.kwargs:
            for resTy in self.kwargs['results']:
                assert(isinstance(resTy, MLIRType))
                self.addResult(resTy)
        self.func = self.kwargs['func']
        self.isGate = 'isGate' in self.kwargs

    def show(self) -> str:
        args = ', '.join(map(lambda a: a.show(), self.operands))
        argty = ', '.join(map(lambda a: a.getType().show(), self.operands))
        resty = ', '.join(map(lambda r: r.show(), self.results))
        if len(self.results) != 1:
            resty = '(' + resty + ')'
        attrs = '{qasm.gate} ' if self.isGate else ''
        return f'@{self.func}({args}) {attrs}: ({argty}) -> {resty}'

class ConstantOp(MLIROperation):
    """Constant Op
    build(val: MLIRAttribute, ty: MLIRType)
    """
    name = 'constant'
    def build(self):
        val, ty = self.args
        self.addAttribute(val)
        self.addResult(ty)
    def getValue(self) -> MLIRAttribute:
        return self.attributes[0]
    def getType(self) -> MLIRType:
        return self.results[0].getType()
    def show(self) -> str:
        return f'{self.getValue()} : {self.getType()}'

class UnaryOp(MLIROperation):
    def operand(self) -> SSAValue:
        return self.operands[0]
    def build(self):
        self.addOperand(self.args[0])
        self.addResult(self.args[0].getType())
    def show(self) -> str:
        return self.operand().show(withType=True)

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

class NegFOp(UnaryOp):
    name = 'negf'

class SIToFPOp(UnaryOp):
    name = 'sitofp'
    def build(self):
        assert(isinstance(self.args[0].getType(), IntType))
        self.addOperand(self.args[0])

        if len(self.args) == 2:
            assert(isinstance(self.args[1], FloatType))
            self.addResult(self.args[1])
        else:
            self.addResult(FloatType())
    def show(self) -> str:
        return f'{self.operand().show(True)} to {self.results[0].getType()}'

class BinaryOp(MLIROperation):
    """build(lhs, rhs)
    """
    def build(self):
        lhs = self.args[0]
        rhs = self.args[1]
        self.addOperand(lhs)
        self.addOperand(rhs)
        lhsty, rhsty = lhs.getType(), rhs.getType()
        assert(type(lhsty) == type(rhsty))
        self.addResult(lhsty)

    def lhs(self) -> SSAValue:
        return self.operands[0]
    def rhs(self) -> SSAValue:
        return self.operands[1]
    def getType(self) -> MLIRType:
        return self.results[0].getType()
    def show(self) -> str:
        return f'{self.lhs()}, {self.rhs()} : {self.lhs().getType()}'

class AddFOp(BinaryOp):
    name = 'addf'
class SubFOp(BinaryOp):
    name = 'subf'
class MulFOp(BinaryOp):
    name = 'mulf'
class DivFOp(BinaryOp):
    name = 'divf'

class AddIOp(BinaryOp):
    name = 'addi'
class SubIOp(BinaryOp):
    name = 'subi'
class MulIOp(BinaryOp):
    name = 'muli'
class DivIOp(BinaryOp):
    name = 'divi'

class CmpIOp(MLIROperation):
    """build(lhs, rhs[, cmpMnemonic])
    """
    def build(self):
        lhs = self.args[0]
        rhs = self.args[1]
        self.addOperand(lhs)
        self.addOperand(rhs)
        lhsty, rhsty = lhs.getType(), rhs.getType()
        assert(type(lhsty) == type(rhsty))
        self.addResult(IntType(1))
        if len(self.args) >= 3:
            self.addAttribute(StringAttr(self.args[2]))
        else:
            self.addAttribute(StringAttr("eq"))

    def lhs(self) -> SSAValue:
        return self.operands[0]
    def rhs(self) -> SSAValue:
        return self.operands[1]
    def getType(self) -> MLIRType:
        return self.results[0].getType()
    def getCmpMnemonic(self) -> MLIRAttribute:
        return self.attributes[0]
    def show(self) -> str:
        return f'{self.getCmpMnemonic()}, {self.lhs()}, {self.rhs()} : {self.lhs().getType()}'
    
""" qasm Ops """
class QASMAllocOp(MLIROperation):
    name = 'allocate'
    dialect = 'qasm'
    def build(self):
        self.addResult(QubitType())
    def show(self) -> str:
        return ''

class PIOp(MLIROperation):
    name = 'pi'
    dialect = 'qasm'
    def getType(self):
        return self.results[0].getType()
    def build(self):
        self.addResult(FloatType())
    def show(self) -> str:
        return f': {self.getType()}'

class UOp(MLIROperation):
    name = 'U'
    dialect = 'qasm'
    """build(theta, phi, lambd, qubit)
    """
    def build(self):
        args = self.args

        for i in range(3):
            self.addOperand(args[i])
            assert(isinstance(args[i].getType(), FloatType))
        self.theta = args[0]
        self.phi = args[1]
        self.lambd = args[2]
        
        self.addOperand(args[3])
        assert(isinstance(args[3].getType(), QubitType))
        self.qubit = args[3]
    def show(self) -> str:
        return f'({self.theta.show(True)}, {self.phi.show(True)}, {self.lambd.show(True)})'                 f' {self.qubit.show()}'

class CNOTOp(MLIROperation):
    name = 'CX'
    dialect = 'qasm'
    """build(q0, q1)
    """
    def build(self):
        args = self.args
        for i in range(2):
            assert(isinstance(args[i].getType(), QubitType))
            self.addOperand(args[i])
        self.control = args[0]
        self.target = args[1]
    def show(self) -> str:
        return f'{self.control}, {self.target}'

class QASMMeasureOp(MLIROperation):
    name = 'measure'
    dialect = 'qasm'
    """build(q)
    """
    def build(self):
        assert(len(self.args) == 1)
        qubit = self.args[0]
        assert(isinstance(qubit.getType(), QubitType))
        self.addOperand(qubit)
        self.addResult(IntType(1))
    def getQubit(self) -> SSAValue:
        return self.operands[0]
    def show(self) -> str:
        return f'{self.getQubit()}'

class SingleQubitConsumeOp(MLIROperation):
    dialect = 'qasm'
    def build(self):
        assert(len(self.args) == 1)
        qubit = self.args[0]
        assert(isinstance(qubit.getType(), QubitType))
        self.addOperand(qubit)
    def getQubit(self) -> SSAValue:
        return self.operands[0]
    def show(self) -> str:
        return f'{self.getQubit()}'

class QASMResetOp(SingleQubitConsumeOp):
    name = 'reset'

class QASMBarrierOp(SingleQubitConsumeOp):
    name = 'barrier'

class QASMIfOp(MLIROperation):
    name = 'if'
    dialect = 'quantum'
    """SCF If Op
    # use args only
    build(creg: SSAValue, val: IntAttr, scope: SSAValueMap)
    // %cond : i1
    qasm.if %cond $val : type(%cond) {
        [[ifBlock]]
    }
    
    Then add ops using the if-block: `<ifOp>.getIfBlock()`
    """
    def build(self):
        creg, val, scope = self.args
        assert(isinstance(creg.getType(), MemrefType))
        self.addOperand(creg)
        self.addAttribute(val)
        self.addBlock(MLIRBlock(scope))

    def getCreg(self) -> SSAValue:
        return self.operands[0]
    def getValue(self) -> MLIRAttribute:
        return self.attributes[0]
    def getIfBlock(self):
        return self.blocks[0]

    def serialize(self) -> list[str]:
        lines = [f'qasm.if {self.getCreg()} = {self.getValue()} : {self.getCreg().getType()} {{']
        lines += self.indent(self.getIfBlock().serialize())
        lines.append('}')
        return lines
    def show(self) -> str:
        return '\n'.join(self.serialize())

### Memref Ops
class AllocOp(MLIROperation):
    """Alloc Op
    build(dims: list[Union[None, int]], ty: MLIRType)
    """
    dialect = 'memref'
    name = 'alloc'
    def getType(self) -> MemrefType:
        return self.results[0].getType()
    def getElementType(self) -> MLIRType:
        return self.getType().getElementType()

    def build(self):
        dims = self.kwargs['dims']
        elemty = self.kwargs['ty']
        ty = MemrefType(dims=dims, ty=elemty)
        self.addResult(ty)
        
        self.dyn_dims = 0
        for d in dims:
            if d is None:
                self.dyn_dims += 1
        if self.dyn_dims > 0:
            raise UnimplementedError("No support for dynamic dims in alloc")

    def show(self) -> str:
        dyn_dims = ''
        return f'({dyn_dims}) : {self.getType()}'

### Affine Ops
class StoreOp(MLIROperation):
    """Store Op
    # use kwargs only
    build(mem: SSAValue, idx: IntAttr, val: SSAValue)
    // %mem : memref<?xi1>
    // %idx : index 
    memref.store %val, %mem[%idx] : memref<?xi1>
    """
    name = 'store'
    dialect = 'affine'
    def build(self):
        args = self.kwargs
        self.addOperand(args['mem'])
        self.addAttribute(args['idx'])
        self.addOperand(args['val'])
        elemTy = args['mem'].getType().getElementType()
        valTy = args['val'].getType()
        assert(type(elemTy) == type(valTy))
    def getMemref(self):
        return self.operands[0]
    def getIndex(self):
        return self.attributes[0]
    def getValue(self):
        return self.operands[1]
    def show(self) -> str:
        return f'{self.getValue()}, {self.getMemref()}[{self.getIndex()}] : {self.getMemref().getType()}'
    
class LoadOp(MLIROperation):
    """Load Op
    # use kwargs only
    build(mem: SSAValue, idx: SSAValue)
    
    // %mem : memref<?xi1>
    // %idx : index
    %res = memref.load %mem[%idx] : memref<?xi1>
    """
    name = 'load'
    dialect = 'affine'
    def build(self):
        args = self.kwargs
        self.addOperand(args['mem'])
        self.addOperand(args['idx'])
        self.addResult(args['mem'].getType().getElementType())
    def getMemref(self):
        return self.operands[0]
    def getIndex(self):
        return self.operands[1]
    def getValue(self):
        return self.results[0]
    def show(self) -> str:
        return f'{self.getMemref()}[{self.getIndex()}] : {self.getMemref().getType()}'


ExpressionType = Union[Node.Id, Node.BinaryOp, Node.Real, Node.Int, Node.Prefix]
def isaExpression(obj) -> bool:
    if isinstance(obj, Node.Id): return True
    if isinstance(obj, Node.BinaryOp): return True
    if isinstance(obj, Node.Real): return True
    if isinstance(obj, Node.Int): return True
    if isinstance(obj, Node.Prefix): return True
    return False

OperationType = Union[Node.UniversalUnitary, Node.CustomUnitary, Node.Cnot, Node.Qreg, Node.Creg, 
                      Node.Measure, Node.Reset, Node.Barrier, Node.If]
def isaOperation(obj) -> bool:
    if isinstance(obj, Node.UniversalUnitary): return True
    if isinstance(obj, Node.CustomUnitary): return True
    if isinstance(obj, Node.Cnot): return True
    if isinstance(obj, Node.Qreg): return True
    if isinstance(obj, Node.Creg): return True
    if isinstance(obj, Node.Measure): return True
    if isinstance(obj, Node.Reset): return True
    if isinstance(obj, Node.Barrier): return True
    if isinstance(obj, Node.If): return True
    return False

class MLIRBlock(MLIRBase):
    def __init__(self, valueMap: SSAValueMap = SSAValueMap()):
        self.valueMap: SSAValueMap = valueMap
        self.body: list[MLIROperation] = []

    def serialize(self) -> list[str]:
        code = []
        for op in self.body:
            code += op.serialize()
        return code

    def addOp(self, op: MLIROperation):
        self.body.append(op)
    def buildOp(self, opClass: type, *args, **kwargs) -> list[SSAValue]:
        op = opClass(self.valueMap, *args, **kwargs)
        self.addOp(op)
        return op.getResults()

    def castToFloat(self, val: SSAValue, ft: FloatType = FloatType()) -> SSAValue:
        if isinstance(val.getType(), FloatType):
            return val
        if isinstance(val.getType(), IntType):
            return self.buildOp(SIToFPOp, val, ft)[0]
        raise ConversionError("Attempting to cast invalid value to float")

    def parseQubit(self, node: Union[Node.Id, Node.IndexedId]) -> list[SSAValue]:
        if isinstance(node, Node.Id):
            return self.valueMap.resolve(node.name)
        elif isinstance(node, Node.IndexedId):
            return [self.valueMap.resolve(node.name)[node.index]]
        else:
            raise UnimplementedError()
    def parseMemref(self, node: Union[Node.Id, Node.IndexedId]) -> list[(SSAValue, int)]:
        if isinstance(node, Node.Id): # full register
            mem = self.valueMap.lookup(node.name)
            sz = mem.getType().getSize(0)
            bits = []
            for i in range(sz):
                bits.append((mem, i))
            return bits
        elif isinstance(node, Node.IndexedId): # single bit
            mem = self.valueMap.lookup(node.name)
            return [(mem, node.index)]
        else:
            raise UnimplementedError()


    def parseExpression(self, node: ExpressionType) -> SSAValue:
        logger.debug(f'>> EXPRESSION: {type(node)} {node.qasm()}')
        if isinstance(node, Node.BinaryOp):
            binop: Node.BinaryOperator = node.children[0]
            lhs = self.parseExpression(node.children[1])
            rhs = self.parseExpression(node.children[2])
            op = None
            if isinstance(lhs.getType(), FloatType) or isinstance(rhs.getType(), FloatType):
                lhs = self.castToFloat(lhs)
                rhs = self.castToFloat(rhs)
                if binop.value == '+': op = AddFOp
                if binop.value == '-': op = SubFOp
                if binop.value == '*': op = MulFOp
                if binop.value == '/': op = DivFOp
            else:
                if binop.value == '+': op = AddIOp
                if binop.value == '-': op = SubIOp
                if binop.value == '*': op = MulIOp
                if binop.value == '/': op = DivIOp
            res = self.buildOp(op, lhs, rhs)
            return res[0]
        if isinstance(node, Node.Id):
            return self.valueMap.lookup(node.name)
        if isinstance(node, Node.Real):
            if node.qasm() == 'pi':
                res = self.buildOp(PIOp, FloatType())
            else:
                res = self.buildOp(ConstantOp, FloatAttr(node.value), FloatType())
            return res[0]
        if isinstance(node, Node.Int):
            res = self.buildOp(ConstantOp, FloatAttr(float(node.value)), FloatType())
            return res[0]
        if isinstance(node, Node.Prefix):
            op, val = node.children
            res = self.parseExpression(val)
            if isinstance(op, Node.UnaryOperator):
                if op.qasm() == '-':
                    if isinstance(res.getType(), FloatType):
                        nres = self.buildOp(NegFOp, res)[0]
                        return nres

        showtree(node)
        raise UnimplementedError(f"Unknown expression kind {type(node)}: {node.qasm()}")

    def parseOperation(self, op: OperationType) -> Union[None, SSAValue]:
        if isinstance(op, Node.UniversalUnitary):
            args = op.children[0].children

            theta = self.parseExpression(args[0])
            theta = self.castToFloat(theta)
            phi   = self.parseExpression(args[1])
            phi   = self.castToFloat(phi)
            lambd = self.parseExpression(args[2])
            lambd = self.castToFloat(lambd)

            qubits = self.parseQubit(op.children[1])
            for qubit in qubits:
                self.buildOp(UOp, theta, phi, lambd, qubit)
            return
        if isinstance(op, Node.Cnot):
            conts = self.parseQubit(op.children[0])
            targs = self.parseQubit(op.children[1])
            if len(conts) > 1 and len(targs) > 1:
                for (cont, targ) in zip(conts, targs):
                    self.buildOp(CNOTOp, cont, targ)
            else:
                for cont in conts:
                    for targ in targs:
                        self.buildOp(CNOTOp, cont, targ)
            return
        if isinstance(op, Node.CustomUnitary):
            params: list[SSAValue] = []
            if op.arguments is not None:
                for arg in op.arguments.children:
                    param = self.parseExpression(arg)
                    param = self.castToFloat(param)
                    params.append(param)

            num = 1
            qubitss: list[list[SSAValue]] = []
            for q in op.bitlist.children:
                qs = self.parseQubit(q)
                if len(qs) != 1: num = len(qs)
                qubitss.append(qs)

            for i in range(num):
                call_args: list[SSAValue] = copy.copy(params)
                for qs in qubitss:
                    call_args.append(qs[0 if len(qs) == 1 else i])
                self.buildOp(CallOp, func=op.name, operands=call_args, results=[], isGate=True)
            return
        if isinstance(op, Node.Qreg):
            name = op.name
            size = op.index
            qubits: list[SSAValue] = [self.buildOp(QASMAllocOp)[0] for _ in range(size)]
            self.valueMap.insertArray(name, qubits)
            return
        if isinstance(op, Node.Creg):
            name = op.name
            size = op.index
            bits: SSAValue = self.buildOp(AllocOp, dims=[size], ty=IntType(1))[0]
            self.valueMap.insert(name, bits)
            return
        if isinstance(op, Node.Measure):
            qubits = self.parseQubit(op.children[0])
            bits = self.parseMemref(op.children[1])
            for (qubit, memLoc) in zip(qubits, bits):
                mem, idx = memLoc
                result = self.buildOp(QASMMeasureOp, qubit)[0]
                self.buildOp(StoreOp, mem=mem, idx=int(idx), val=result)
            return
        if isinstance(op, Node.Reset):
            qubits = self.parseQubit(op.children[0])
            for qubit in qubits:
                self.buildOp(QASMResetOp, qubit)
            return
        if isinstance(op, Node.Barrier):
            for arg in op.children[0].children:
                qubits = self.parseQubit(arg)
                for qubit in qubits:
                    self.buildOp(QASMBarrierOp, qubit)
            return
        if isinstance(op, Node.If):
            creg, val, childOp = op.children
            mem = self.valueMap.lookup(creg.name)
            self.parseOperation(childOp)
            qInst = self.body.pop() # extract last inst, place it inside if block
            self.buildOp(QASMIfOp, mem, IntAttr(val.value), self.valueMap)
            self.body[-1].getIfBlock().body.append(qInst)
            return

        showtree(op)
        raise UnimplementedError(f"Unknown operation kind {type(op)}: {op.qasm()}")

class MLIRFunction(MLIRBase):
    def __init__(self, name: str):
        self.valueMap: SSAValueMap = SSAValueMap()
        self.name: str = name
        self.arguments: list[SSAValue] = []
        self.results: list[MLIRType] = []
        self.body: MLIRBlock = MLIRBlock(self.valueMap)
        self.attributes: list[str] = []
        self.private = False
        self.hasBody = True

    def addArgument(self, name: str, ty: MLIRType):
        arg = self.valueMap.newValue(ty, name)
        self.arguments.append(arg)
    def addResult(self, ty: MLIRType):
        self.results.append(ty)
    def addAttribute(self, attr: str):
        self.attributes.append(attr)
    
    def setPrivate(self):
        self.private = True

    def serialize(self) -> list[str]:
        args = ', '.join(map(lambda v: v.show(True), self.arguments))
        results = ', '.join(map(lambda t: t.show(), self.results))
        attr_list = ''
        if len(self.attributes) > 0:
            attr_list = 'attributes {' + ', '.join(self.attributes) + '}'
        privateStr = ('private' if self.private else '')
        code = [f'func {privateStr} @{self.name} ({args}) -> ({results}) {attr_list}']
        if self.hasBody:
            code.append('{')
            code += self.indent(self.body.serialize())
            code.append('}')
        return code

qasm_stdgates: list[str] = 'u3 u2 u1 cx id u0 u p x y z h s sdg t tdg rx ry rz sx sxdg cz cy swap ch ccx cswap crx cry crz cu1 cp cu3 csx cu rxx rzz rccx rc3x c3x c3sqrtx c4x'.split() 
class MLIRModule(MLIRBase):
    def __init__(self, strict=False):
        self.declarations: list[MLIRFunction] = []
        self.strict = strict

    def serialize(self) -> list[str]:
        code = ['module {']
        for decl in self.declarations:
            code += self.indent(decl.serialize())
        code.append('}')
        return code

    def addDecl(self, decl):
        self.declarations.append(decl)

    def addFunction(self, name: str) -> MLIRFunction:
        func = MLIRFunction(name)
        self.declarations.append(func)
        return func

    def parseGate(self, node: Union[Node.Gate, Node.Opaque]):
        gate = self.addFunction(node.name)
        gate.addAttribute('qasm.gate')
        gate.setPrivate()
        if node.name in qasm_stdgates:
            gate.addAttribute(f'qasm.stdgate="{node.name}"')
        if node.arguments is not None:
            for arg in node.arguments.children:
                gate.addArgument(arg.name, FloatType())
        for arg in node.bitlist.children:
            gate.addArgument(arg.name, QubitType())
        if isinstance(node, Node.Opaque):
            gate.hasBody = False
        else:
            for op in node.body.children:
                gate.body.parseOperation(op)
        gate.body.buildOp(ReturnOp, isGate=True)

    def parseVersion(self, version: Node.Format):
        if self.strict:
            raise UnimplementedError("Version String")


def QASMToMLIR(code: str, strict=False) -> MLIRModule:
    try:
        src = Qasm(data=code).parse()
    except:
        raise ConversionError("Could not parse QASM")

    module: MLIRModule = MLIRModule(strict=strict)
    mainFunc: MLIRFunction = MLIRFunction('qasm_main')
    mainFunc.addAttribute('qasm.main')
    for node in src.children:
        logger.debug(f'>> PARSING:\n {node.qasm()}\n<<<<<<<<')
        if isinstance(node, Node.Format):
            module.parseVersion(node)
        elif isinstance(node, Node.Gate):
            module.parseGate(node)
        elif isinstance(node, Node.Opaque):
            module.parseGate(node)
        elif isaOperation(node):
            mainFunc.body.parseOperation(node)
        else:
            raise ConversionError(f"Unknown node object of type {type(node)} found: {node.qasm()}")
    mainFunc.body.buildOp(ReturnOp)
    module.addDecl(mainFunc)

    return module


def main():
    parser = argparse.ArgumentParser(description='QASM to MLIR translation tool')
    parser.add_argument('-i', metavar='input', dest='input', type=str,
            help='Input file (uses stdin if not specified)', required=False)
    parser.add_argument('-o', metavar='output', dest='output', type=str,
            help='Output file (uses stdout if not specified)', required=False)
    parser.add_argument('-v', action='store_true', dest='verbose',
            help='verbose', required=False)
    parser.add_argument('--config', metavar='config_file', dest='config', type=str,
            help='Configuration file to use (overrides other cmdline options). Each line in the config file should be of the form <input-file>,<output-file>', required=False)
    args = parser.parse_args()

    if args.config is not None:
        # parse config file and use that instead
        # each line should be of the form:
        # <input-file>,<output-file>
        with open(args.config) as configFile:
            for line in configFile.readlines():
                ipname, opname = line.strip().split(',')
                print(f'Converting {ipname} -> {opname}')
                with open(ipname, 'r') as ipf:
                    code = ipf.read()
                    module = QASMToMLIR(code, strict=False)
                    with open(opname, 'w+') as opf:
                        opf.write(str(module))
        return

    if args.input is None: args.input = sys.stdin
    else: args.input = open(args.input, 'r')
    if args.output is None: args.output = sys.stdout
    else: args.output = open(args.output, 'w+')

    code = args.input.read()
    module = QASMToMLIR(code, strict=args.verbose)
    args.output.write(str(module))


if __name__ == "__main__": main()
