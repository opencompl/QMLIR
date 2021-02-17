#! env python3

# OpenQASM Lang Spec: https://arxiv.org/pdf/1707.03429v2.pdf

import argparse
import copy
from qiskit.qasm import Qasm

class ConversionError(Exception):
    pass

class ModuleWriter:
    def __init__(self, output_file):
        self.currentIndex = 0
        self.output_file = output_file
        module = []
        pass

    def addFunction(self, func):
        module.append(func)
    def writeOp(self, op, numResults=1):
        results = None
        if numResults == 1:
            self.output_file.write('%{} = {}'.format(self.currentIndex, op))
            results = '%{}'.format(self.currentIndex)
        else:
            self.output_file.write('%{}:{} = {}'.format(self.currentIndex, numResults, op))
            results = ['%{}#{}'.format(self.currentIndex, i) for i in range(numResults)]
        self.currentIndex += 1
        return results

    def flush(self):
        for decl in self.module:
            for inst in decl:
                output_file.write(inst + '\n')

class SSAValueMap:
    def __init__(self, moduleWriter):
        self.store = dict()
        self.moduleWriter = moduleWriter

    def addQreg(self, name, size):
        values = []
        for i in range(size):
            value = moduleWriter.writeOp('qssa.allocate() : !qssa.qubit<1>')
            values.append(value)
        self.store[name] = values

def QASMToMLIR(input_file, output_file, strict=False):
    src = Qasm(input_file).parse()
    print (src)


def main():
    parser = argparse.ArgumentParser(description='QASM to MLIR translation tool')
    parser.add_argument('-o', type=str, help='output file', required=False)
    args = parser.parse_args()
    print("NO")

if __name__ == "__main__":
    main()
