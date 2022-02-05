QSSA Artifact
=============

Created for the submission at "ACM SIGPLAN 2022 International Conference on Compiler Construction (CC-2022)".

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

Executing and generating plots.  

Steps:
1. `cd` into the dataset folder.
2. Place all your `.qasm` programs in `./circuit_qasm/`.
3. Copy all python scripts and the Makefile from `/artifact/QMLIR/data/scripts/`. (excluding `plot-bench-runtimes-ibmchallenge.py`, unless running on the IBM dataset).
4. `mkdir circuit_mlir`
5. [optional] `make clean`: removes generated output programs and benchmark data/plots, if any.
6. `make plots`: run all the benchmarks and generate plots.

The two datasets from the paper are present in `/artifact/QASMBench` and `/artifact/IBMChallenge`. The `.qasm` programs, plotting scripts and Makefile for these are already extracted and placed correctly.

The plots are generated in the same folder as PDFs.

Detailed Usage
--------------

See the `Makefile` for exact targets and workflow.
