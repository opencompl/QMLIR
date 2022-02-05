#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import sys
import copy
import math
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

parser = argparse.ArgumentParser(description="""[QSSA] Tool to plot comparative runtimes of optimizations for the IBM Challenge dataset.
Splits the dataset into two parts, <= 6000 lines and > 6000 lines; and plots separately.
Works only on the IBM set.
""")
parser.add_argument('-i', metavar='datafile', dest='datafile', type=str,
    help='generated JSON data file', required=True)
outfile_default = os.path.basename(__file__).replace(".py", ".pdf")
parser.add_argument('-o', metavar='outfile', dest='outfile', type=str,
    help=f'output pdf file name (defaults to {outfile_default})', required=False)
args = parser.parse_args()
if args.outfile is None: args.outfile = outfile_default

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
with open(args.datafile, "r") as f:
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
            # if kind == 'qssa':
            #    self.time -= stats['passes']['Inliner']

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

#### PLOTTING-CODE
log(">> Plotting [%d] test cases..."% (len(plotdata)))


fig, axs = plt.subplots(1, 2, figsize=(15,10), gridspec_kw={'width_ratios': [6,1], 'wspace': 0.20, 'left':0.10})
for plot_idx in [0, 1]:
    ax = axs[plot_idx]
    if plot_idx == 0:
        to_plot = plotdata[:133]
    else:
        to_plot = plotdata[133:]

    lo, hi = 10**9, -1
    for p in to_plot:
        lo = min(lo, p.getKind('default').tot)
        hi = max(hi, p.getKind('default').tot)
    log(f">> Plot {plot_idx}: range = ({lo}, {hi})")
    xs = np.arange(len(to_plot))
    width = 0.35
    for idx, kind in enumerate(['qiskit_lev1', 'qssa']):
        ratio = lambda p: p.getKind(kind).time
        ratio = lambda p: p.getKind('qiskit_lev2').time / p.getKind(kind).time

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
        if kind == 'qssa':
            col = dark_blue
            label = 'qssa'
        rects1 = ax.bar(xs + ((idx + 1) * 1 * width), [ratio(p) for p in to_plot], width, label=label, color=col)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if plot_idx == 1: # only for right side
        ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0), fontsize=LABEL_FONT_SIZE)
        ax.set_yticks([5,10,15,20])
    else: # only for left side
        ax.set_ylabel('speedup over qiskit -O2', rotation='horizontal', position = (1, 1.05),
            horizontalalignment='left', verticalalignment='bottom', fontsize=LABEL_FONT_SIZE)
        ax.set_yticks([1,2,3,4])

    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(['$\\leq$6000 gates', '>6000 gates'][plot_idx], fontsize=LABEL_FONT_SIZE)


fig.set_size_inches(5,2)
fig.tight_layout()
fig.savefig(args.outfile)
log(f'output plot written to {args.outfile}')
