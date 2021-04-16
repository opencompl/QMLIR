#!/usr/bin/env python3
import argparse
import glob
import os
import sh
import json
import sys

parser = argparse.ArgumentParser(description='Run MLIR files to document number of thunks dropped')
args = parser.parse_args()

g_opt = sh.quantum_opt
g_fpaths = glob.glob("circuit_mlir/*.qasm.mlir")

g_data = []
for (i, fpath) in enumerate(g_fpaths):
    print("[%3d/%3d] |%60s|" % (i+1, len(g_fpaths), fpath))
    call = g_opt(fpath, "--qasm-gate-count", "-o", "/dev/null").wait()
    stderr = call.stderr.decode()
    g_data.append(json.loads(stderr))

with open("gate-count-data.json", "w") as f:
    json.dump(g_data, f, indent=2)
