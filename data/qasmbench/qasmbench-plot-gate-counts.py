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
with open("./qasmbench-gate-count.json", "r") as f:
    rawdata = json.load(f)
assert rawdata is not None

INF = 10**9
class PlotDatum:
    def __init__(self, label, kind, stats, idx):
        self.label = label # testname
        self.kind = kind # default, qiskit, qssa, zx
        self.idx = idx
        if 'ops' not in stats or 'depth' not in stats:
            log(f'> INVALID {label}::{kind} : {stats}')
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
pidx = 0
for test in rawdata:
    pidx += 1
    data = FullData(test, rawdata[test], pidx)
    plotdata.append(data)
    # if data.getKind('qssa_full').tot < data.getKind('qiskit_lev2').tot:
    #     to_show = {'name': test, 'qssa': rawdata[test]['qssa_full'], 'qiskit':rawdata[test]['qiskit_lev2']}
    #     print(json.dumps(to_show, indent=2))
plotdata.sort()


#### PLOTTING-CODE
to_plot = plotdata
log(">> Plotting [%d] test cases..."% (len(to_plot)))

xs = np.arange(len(to_plot))
width = 0.2

#### Optimization ratio
fig, ax = plt.subplots(figsize=(15,10))
for idx, kind in enumerate(['qiskit_lev1', 'qiskit_lev2', 'qiskit_lev3', 'qssa_full']):
    ratio = lambda p: 100*(1-p.getKind(kind).tot / p.getKind('default').tot)
    col = None
    label = None
    if kind == 'qiskit_lev1':
        col = light_green
        label = 'qiskit -O1'
    if kind == 'qiskit_lev2':
        col = dark_green
        label = 'qiskit -O2'
    if kind == 'qiskit_lev3':
        col = light_blue
        label = 'qiskit -O3'
    if kind == 'qssa_full':
        col = dark_blue
        label = 'qssa'
    rects1 = ax.bar(xs + ((idx + 1) * 1 * width), [ratio(p) for p in to_plot], width, label=label, color=col)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0), fontsize=LABEL_FONT_SIZE)

ax.set_xticks([])
ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
ax.set_ylabel('%optimization', rotation='horizontal', position = (1, 1.05),
    horizontalalignment='left', verticalalignment='bottom', fontsize=LABEL_FONT_SIZE)

fig.set_size_inches(5,2)
fig.tight_layout()
filename = os.path.basename(__file__).replace(".py", ".pdf")
fig.savefig(filename)

### Stats for the paper
beat1, equal1, fail1 = [], [], []
beat2, equal2, fail2 = [], [], []
for lev in [1,2,3]:
    beat = []
    equal = []
    fail = []
    for p in to_plot:
        qssa = p.getKind('qssa_full')
        qis = p.getKind('qiskit_lev' + str(lev))
        if qssa.tot < qis.tot:
            beat.append(p.test)
        elif qssa.tot == qis.tot:
            equal.append(p.test)
        else:
            fail.append(p.test)
    print(f">>>>>>>>> LEVEL {lev} >>>>>>>>>>>>")
    print(f'> beat = {len(beat)}, equal = {len(equal)+len(beat)}')
    print()
    print(f'> beat: {beat}')
    print()
    print(f'> equal: {equal}')
    print()
    print(f'> fail: {fail}')
    print()
    print("---------------------------------")
