from __future__ import annotations
import os, glob, copy, timeit
import argparse
import sh, json

import qiskit
import qiskit.test.mock

class UnimplementedError(Exception):
    pass

""" Single Base Tool to run an optimizer """
class Opt:
    def __init__(self, filename, **kwargs):
        self.circuit = None
        self.no_opt = False # enable optimization
        if 'no_opt' in kwargs:
            self.no_opt = kwargs['no_opt']
        self.input_args = kwargs
        self.load(filename)

    def load(self, filename):
        """ load circuit from qasm file """
        raise UnimplementedError()
    def opt(self):
        """ optimize circuit and cache it """
        raise UnimplementedError()
    def qasm(self):
        """ return optimized qasm program (if possible) """
        raise UnimplementedError()

    def getStats(self):
        """ Return optimized circuit info """
        raise UnimplementedError()

class QiskitOpt(Opt):
    def load(self, filename):
        self.filename = filename
        self.circuit = qiskit.circuit.QuantumCircuit.from_qasm_file(filename)
        if 'opt_level' in self.input_args:
            self.opt_level = int(self.input_args['opt_level'])
        else:
            self.opt_level = 3
        if self.no_opt:
            self.opt_level = 0

    def opt(self):
        backend = qiskit.test.mock.FakeQasmSimulator()
        self.circuit_opt = qiskit.transpile(self.circuit, backend, optimization_level=self.opt_level)

    def getStats(self):
        qopt = self.circuit_opt.decompose()
        return {'depth': qopt.depth(), 'ops': qopt.count_ops()}

quantum_opt = sh.Command("quantum-opt")
class QSSAOpt(Opt):
    def load(self, filename):
        """
        loads circuit_mlir/<file>.qssa.mlir
        runs make to build the file if not present
        """

        self.filename = filename.replace('circuit_qasm', 'circuit_mlir').replace('.qasm', '.qssa.mlir')
        sh.make(self.filename)

        self.outfile = filename.replace('circuit_qasm', 'circuit_mlir').replace('.qasm', '.qssa.opt.mlir')
        self.taskname = os.path.basename(filename).replace('.qasm', '')

        self.data = None

    def opt(self):
        quantum_opt(self.filename,
                    '--inline',
                    '--cse',
                    '--symbol-dce',
                    '--qssa-apply-rewrites',
                    '--qssa-convert-1q-to-U',
                    '--qssa-apply-rewrites',
                    _out=self.outfile)

    def getStats(self):
        quantum_opt(self.outfile,
                    '--canonicalize',
                    '--qssa-compute-depths',
                    '--qssa-gate-count',
                    _out="/dev/null",
                    _err=f".{self.taskname}.tmp.json")
        with open(f".{self.taskname}.tmp.json") as f:
            self.data = json.load(f)
        if "qasm_main" not in self.data:
            return {'depth': 0, 'ops': {}}
        return self.data["qasm_main"]

""" Benchmark one program """
class BenchmarkOne:
    def __init__(self, filename):
        self.opts = dict()
        with open(filename, 'r') as f:
            self.prog = f.read()

        self.opts['default'] = QiskitOpt(filename, no_opt=True)
        self.opts['qiskit_lev1'] = QiskitOpt(filename, opt_level=1)
        self.opts['qiskit_lev2'] = QiskitOpt(filename, opt_level=2)
        self.opts['qiskit_lev3'] = QiskitOpt(filename, opt_level=3)
        self.opts['qssa'] = QSSAOpt(filename)
        ### add more opt tools if needed

        self.times = dict()

    def run(self):
        for k in self.opts:
            tm = timeit.timeit(lambda: self.opts[k].opt(), number=1)
            self.times[k] = tm

    def stats(self):
        stats = dict()
        for k in self.opts:
            stat = self.opts[k].getStats()
            stat['time'] = self.times[k]
            stats[k] = copy.deepcopy(stat)
        self.stats = stats
        return stats

    def qasm(self):
        ### display all output qasm programs
        sep_width = 80
        print(self.paddedStr('>>>>> INPUT PROGRAM ', sep_width, '>'))
        print(self.prog)
        print(self.paddedStr('<<<<< INPUT PROGRAM ', sep_width, '<'))
        print()
        for k in self.opts:
            print(self.paddedStr(f'----- {k} ', sep_width, '-'))
            print(self.opts[k].qasm())
            print('-' * sep_width)
            print()
    def paddedStr(self, pref, tot, ch):
        return pref + ch * max(0, tot - len(pref))

def runBench():
    files = glob.glob('./circuit_qasm/*.qasm')
    filesWithNumLines = []
    for f in files:
        filesWithNumLines.append((int(sh.wc(f, '-l').split()[0]), f))
    filesWithNumLines.sort()
    print(f'> Running {len(files)} tests...')

    all_stats = {}
    for idx, fnameAndLines in enumerate(filesWithNumLines):
        lines, fname = fnameAndLines
        testName = os.path.basename(fname).replace('.qasm', '')
        print(f'[{idx}] Running {fname} ({lines} lines)...')

        bb = BenchmarkOne(fname)
        bb.run()
        all_stats[testName] = bb.stats()
    return all_stats

info="""Runs the benchmark all .qasm files in `./circuit_qasm/`.\n
Place all your qasm programs in folder `./circuit_qasm/` and invoke this script.\n
For each file, it runs qiskit -O1, -O2, -O3; and qssa's opt tool `quantum_opt`.\n
- To change list of opts run, check class `BenchmarkOne`\n
- To add more opt tools, inherit from class `Opt`, and override the neccessary functions.\n
"""
def main():
    parser = argparse.ArgumentParser(description='QSSA Project Benchmark tool', epilog=info)
    parser.add_argument('-o', metavar='output', dest='output', type=str,
            help='Output JSON file (uses `gate-count-bench.json` if not specified)', required=False)
    parser.add_argument('-p', metavar='program', dest='program', type=str,
            help='.qasm program to run. (if not specified, all files ./circuit_qasm/*.qasm are run)', required=False)
    args = parser.parse_args()
    if args.output is None: args.output = 'gate-count-bench.json'

    if args.program is not None:
        # run one program
        bb = BenchmarkOne(args.program)
        bb.run()
        results = bb.stats()
        print(json.dumps(results, indent=2))
        with open(args.output, 'w+') as f:
            json.dump(results, f, indent=2)
    else:
        # run all programs
        all_stats = runBench()
        with open(args.output, 'w+') as f:
            json.dump(all_stats, f, indent=2)

if __name__ == "__main__": main()
