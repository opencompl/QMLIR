#!/usr/bin/env python3

import os
import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Use TrueType fonts instead of Type 3 fonts
# Type 3 fonts embed bitmaps and are not allowed in camera-ready submissions
# for many conferences. TrueType fonts look better and are accepted.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['figure.figsize'] = 5, 2


# Color palette
light_gray = "#cacaca"
dark_gray = "#827b7b"
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
light_red = "#fb9a99"
dark_red = "#e31a1c"


rawdata = None # raw data
with open("gate-count-data.json", "r") as f:
    rawdata = json.load(f)

assert rawdata is not None


class PlotDatum:
    def __init__(self, label, tot):
        self.label = label
        self.tot = tot

    def __lt__(self, other):
        return self.tot < other.tot

plotdata = [] # data to be plotted

for i, data in enumerate(rawdata):
    # tot_num_gates.append(tot)
    tot = data["gates"]["cnot"] + data["gates"]["u"]
    lbl = os.path.basename(data["path"]).replace(".qasm.mlir", "")
    plotdata.append(PlotDatum(lbl, tot))

plotdata.sort(reverse=True)

x = np.arange(len(plotdata))  # the label locations
width = 10  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x*width - width/2, [d.tot for d in plotdata], width, label='baseline', color = light_blue)
# rects2 = ax.bar(x + width/2, optims, width, label='Optimised', color = dark_blue)

# Y-Axis Label
#
# Use a horizontal label for improved readability.
ax.set_ylabel('# of gates', rotation='horizontal', position = (1, 1.05),
    horizontalalignment='left', verticalalignment='bottom')

# X axis
ax.set_xticks([])
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xticks(x)
# ax.set_xticklabels([d.label for d in plotdata], rotation=90)
ax.legend(ncol=100, frameon=False, loc='lower right', bbox_to_anchor=(0, 1, 1, 0))


# Hide the right and top spines
#
# This reduces the number of lines in the plot. Lines typically catch
# a readers attention and distract the reader from the actual content.
# By removing unnecessary spines, we help the reader to focus on
# the figures in the graph.
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 1 points vertical offset
                    textcoords="offset points",
                    fontsize="smaller",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

filename = os.path.basename(__file__).replace(".py", ".pdf")
fig.set_size_inches(5,2)
fig.savefig(filename)
