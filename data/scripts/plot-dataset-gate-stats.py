#!/usr/bin/env python3

import copy
import os
import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import argparse

parser = argparse.ArgumentParser(description="""[QSSA] Tool to plot gate statistics of the dataset:
histograms of #programs vs log(gate count) and #programs vs. log(circuit depth)""")
parser.add_argument('-i', metavar='datafile', dest='datafile', type=str,
    help='generated JSON data file', required=True)
outfile_default = os.path.basename(__file__).replace(".py", ".pdf")
parser.add_argument('-o', metavar='outfile', dest='outfile', type=str,
    help=f'output pdf file name (defaults to {outfile_default})', required=False)
args = parser.parse_args()
if args.outfile is None: args.outfile = outfile_default


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 6

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

rawdata = None # raw data
with open(args.datafile, "r") as f:
    rawdata = json.load(f)

assert rawdata is not None


# In[ ]:
# Color palette
light_gray = "#cacaca"
dark_gray = "#827b7b"
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
light_red = "#fb9a99"
dark_red = "#e31a1c"



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

plotdata.sort()

to_plot = plotdata

#### Optimization ratio per unit time
fig, axs = plt.subplots(1, 2, figsize=(15,10))
num_bins = 30
axs[0].hist([np.log2(d.getKind('default').tot) for d in to_plot], 
        num_bins, cumulative=True, color=light_blue)
avg_gate_count_log = np.average([np.log10(d.getKind('default').tot) for d in to_plot])
avg_gate_count = np.average([d.getKind('default').tot for d in to_plot])
median_gate_count = np.median([d.getKind('default').tot for d in to_plot])
print("average (log gate count): %f" % (avg_gate_count_log))
print("average gate count: %f" % (avg_gate_count))
print("median gate count: %f" % (median_gate_count))
print("log(average gate count): %f" % (np.log10(avg_gate_count)))

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].locator_params(axis="x", nbins=5)
axs[0].locator_params(axis="y", nbins=10)

# ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0))

# ax.set_xticks([])
axs[0].tick_params(axis='x', labelsize=TICK_FONT_SIZE)
axs[0].tick_params(axis='y', labelsize=TICK_FONT_SIZE)
axs[0].set_xlabel('log(# of gates)', fontsize=LABEL_FONT_SIZE)
axs[0].set_ylabel('# of programs', rotation='horizontal', position = (1, 1.05),
    horizontalalignment='left', verticalalignment='bottom', fontsize=LABEL_FONT_SIZE)

# ========


axs[1].hist([np.log10(d.getKind('default').depth) for d in to_plot], 
        num_bins, cumulative=True, color=light_blue)
avg_gate_depth_log = np.average([np.log10(d.getKind('default').depth) for d in to_plot])
avg_gate_depth = np.average([d.getKind('default').depth for d in to_plot])
median_gate_depth = np.median([d.getKind('default').depth for d in to_plot])
print("average (log gate depth): %f" % (avg_gate_depth_log))
print("average gate depth: %f" % (avg_gate_depth))
print("median gate depth: %f" % (median_gate_depth))
print("log(average gate depth): %f" % (np.log10(avg_gate_depth)))

axs[1].locator_params(axis="y", nbins=10)
axs[1].locator_params(axis="x", nbins=6)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0))

# ax.set_xticks([])
axs[1].tick_params(axis='x', labelsize=TICK_FONT_SIZE)
axs[1].tick_params(axis='y', labelsize=TICK_FONT_SIZE)
axs[1].set_xlabel('log(circuit depth)', fontsize=LABEL_FONT_SIZE)
axs[1].set_ylabel('# of programs', rotation='horizontal', position = (1, 1.05),
    horizontalalignment='left', verticalalignment='bottom', fontsize=LABEL_FONT_SIZE)


# ========

fig.set_size_inches(5,2)
fig.tight_layout()
fig.savefig(args.outfile)
log(f'output plot written to {args.outfile}')