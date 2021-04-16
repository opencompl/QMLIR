#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import sys
import copy
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

LABEL_FONT_SIZE = 7
TICK_FONT_SIZE = 6

# Use TrueType fonts instead of Type 3 fonts
# Type 3 fonts embed bitmaps and are not allowed in camera-ready submissions
# for many conferences. TrueType fonts look better and are accepted.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['figure.figsize'] = 5, 2

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Color palette
light_gray = "#cacaca"
dark_gray = "#827b7b"
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
light_red = "#fb9a99"
dark_red = "#e31a1c"

# math
lg2 = lambda n: math.log(n, 2)


rawdata = None # raw data
with open("./gate-count-bench.json", "r") as f:
    rawdata = json.load(f)
assert rawdata is not None

INF = 10**9
class PlotDatum:
    def __init__(self, label, kind, stats, idx):
        self.label = label # testname
        self.kind = kind # default, qiskit, qssa, zx
        self.idx = idx
        if 'ops' not in stats or 'depth' not in stats:
            print(f'> INVALID {label}::{kind} : {stats}')
            self.gates = {}
            self.cx = -1
            self.u = -1
            self.single_qubit = -1
            self.depth = -1
            self.tot = -1
            self.time = stats.get('time', -1)
        else:
            gates = stats['ops']
            self.gates = copy.deepcopy(gates) # gates dict: {<gate>: <count>, ...}
            self.cx = gates.get("cx", 0)
            self.u = gates.get("u", 0) + gates.get("u3", 0)
            self.single_qubit = 0
            for gate in "h x y z rx ry rz s sdg t tdg u u1 u2 u3".split():
                self.single_qubit += gates.get(gate, 0)
            self.depth = stats['depth']
            self.tot = sum([gates[g] for g in gates])
            self.time = stats['time']
            if kind == 'qssa_full':
                self.time -= stats['passes']['Inliner']

    def __lt__(self, other):
        return self.tot < other.tot
class FullData:
    def __init__(self, test, stats, idx):
        self.test = test
        self.idx = idx
        self.data = dict()
        for kind in rawdata[test]:
            self.data[kind] = PlotDatum(test, kind, stats[kind], idx)
    def __lt__(self, other):
        lv = self.data['default'].tot
        rv = other.data['default'].tot
        if lv != rv: return lv < rv
        lv = self.data['default'].depth
        rv = self.data['default'].depth
        if lv != rv: return lv < rv
        return True
    def hasKind(self, k):
        return k in self.data
    def getKind(self, k):
        return self.data[k]

plotdata = []
plotdata_routing = []
pidx = 0
for test in rawdata:
    pidx += 1
    data = FullData(test, rawdata[test], pidx)
    if test.find('onlyCX') >= 0:
        plotdata_routing.append(data)
    else:
        plotdata.append(data)
plotdata.sort()

### stats for paper
log()
lfilter = lambda f, xs: list(filter(f, xs))
for plot_idx in [0, 1]:
    log()
    if plot_idx == 0:
        to_plot = plotdata[:133]
    else:
        to_plot = plotdata[133:]
    log("> Stats: " + ("small" if plot_idx == 0 else "large"))
    for lev in [1,2]:
        fracs = []
        for p in to_plot:
            qssa = p.getKind('qssa_full')
            qis = p.getKind('qiskit_lev' + str(lev))
            fracs.append(qis.time/qssa.time)
        fracs.sort()
        better = 100*len(lfilter(lambda x: x > 1, fracs)) / len(fracs)
        twice = 100*len(lfilter(lambda x: x > 2, fracs)) / len(fracs)
        thrice = 100*len(lfilter(lambda x: x > 3, fracs)) / len(fracs)
        fourtimes = 100*len(lfilter(lambda x: x > 4, fracs)) / len(fracs)
        fivetimes = 100*len(lfilter(lambda x: x > 5, fracs)) / len(fracs)
        tentimes = 100*len(lfilter(lambda x: x > 10, fracs)) / len(fracs)
        log(f'> level {lev}: faster - {better}; 2x - {twice}; 3x - {thrice}; 4x - {fourtimes}')
        log(f'                                  5x - {fivetimes}; 10x - {tentimes};')



