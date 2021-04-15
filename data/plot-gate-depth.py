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
with open("gate-count-bench.json", "r") as f:
    rawdata = json.load(f)
assert rawdata is not None

INF = 10**9
plotdata = []
plotdataOnlyCX = []
pidx = 0
for test in rawdata:
    pidx += 1
    data = FullData(test, rawdata[test], pidx)
    if test.find('onlyCX') >= 0:
        plotdataOnlyCX.append(data)
    else:
        plotdata.append(data)

plotdata.sort()
plotdataOnlyCX.sort()

plotdata_simple = []
for d in plotdata:
    if d.getKind('qiskit').tot / d.getKind('default').tot < .8:
        print('> anamoly: ' + d.test)
    else:
        plotdata_simple.append(d)

log(">> Plotting [%d] test cases without anamolies!"% (len(plotdata_simple)))
to_plot = plotdata_simple
to_plot = to_plot[0:100]

xs = np.arange(len(to_plot))
width = 1



#### Gate depth
fig, ax = plt.subplots(figsize=(15,10))
col = light_blue
# rects1 = ax.bar(xs - width , [p.getKind('qssa').depth / p.getKind('qiskit').depth for p in to_plot], width, label='gate depth', color=col)

#### Optimization ratio
fig, ax = plt.subplots(figsize=(15,10))

for idx, kind in enumerate(['default', 'qiskit', 'qssa']):
    col = None
    if kind == 'default': col = light_gray
    if kind == 'qiskit': col = dark_blue
    if kind == 'qssa': col = light_blue
    assert col is not None
    rects1 = ax.bar(xs - width + (idx * 1/5), [p.getKind(kind).depth for p in to_plot], 1/5, label=kind, color=col)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0))

ax.set_xticks([])
ax.set_ylabel('gate depth', rotation='horizontal', position = (1, 1.05),
    horizontalalignment='left', verticalalignment='bottom')

fig.set_size_inches(5,2)
fig.tight_layout()
fig.savefig('gate-depths.pdf')

# #### Optimization ratio per unit time
# fig, ax = plt.subplots(figsize=(15,10))
# for idx, kind in enumerate(['qiskit', 'qssa']):
#     # ratio = lambda p: 100*(1-p.getKind(kind).tot / p.getKind('default').tot)
#     # ratio = lambda p: lg2(p.getKind(kind).time + 1)
#     ratio = lambda p: (1 - p.getKind(kind).tot / p.getKind('default').tot) / p.getKind(kind).time
#     col = ''
#     if kind == 'qiskit': col = light_gray
#     if kind == 'qssa': col = dark_blue
#     rects1 = ax.bar(xs - width/2 + (idx*1 * width), [ratio(p) for p in to_plot], width, label=kind, color=col)
#     # ax.plot(xs, [lg2(p.getKind(kind).tot) for p in to_plot], label=kind)
# 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# 
# ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0))
# 
# ax.set_xticks([])
# ax.set_ylabel('%opt/time', rotation='horizontal', position = (1, 1.05),
#     horizontalalignment='left', verticalalignment='bottom')
# 
# fig.set_size_inches(5,2)
# fig.tight_layout()
# fig.savefig('opt-fact-per-unit-time.pdf')
# 
