QSSA Artifact
=============

Created for the submission at "ACM SIGPLAN 2022 International Conference on Compiler Construction (CC-2022)".

DOI: TBA

Tools
------

1. QSSA optimizer: `/artifact/QMLIR/build/bin/quantum-opt`
2. QSSA translate tool: `/artifact/QMLIR/build/bin/quantum-translate`
3. QASM to QSSA converter: `/artifact/QMLIR/tools/openqasm-to-mlir.py`
4. Benchmarking tools (in `/artifact/QMLIR/data/scripts/`):
    - `Makefile`: given source files in `./circuit_qasm/`, generates all MLIR files.
    - `run-opts-bench.py`: runs the benchmark on all files in `./circuit_qasm/`.
5. Plotting tools (in `/artifact/QMLIR/data/scripts/`):
    - `plot-dataset-gate-stats.py`: Gate statistics of the quantum (qasm) program dataset.
    - `plot-bench-gate-stats.py`: Comparative plot of gate counts of optimized programs.
    - `plot-bench-runtimes.py`: Comparative plot of runtimes of optimized programs.


Usage
-----

Executing and generating plots. `cd` into the dataset folder and run:
```
mkdir circuit_mlir
make all
python3 run-opts-bench.py -o results.json
python3 plot-dataset-gate-stats.py -i results.json
python3 plot-bench-gate-stats.py -i results.json
python3 plot-bench-runtimes.py -i results.json
```
