# Requirements:
# 1. make sure `quantum-opt` is in PATH
# 2. make sure `quantum-translate` is in PATH
# 3. set variable `OPENQASM_TO_MLIR_PY` to /path/to/openqasm-to-mlir.py script
# 4. All source programs are in ./circuit_qasm/
# 5. mkdir circuit_mlir
# 6. [optional] to use `make convert_all`, ensure file `convert.txt` contains all programs in CSV format. check `openqasm-to-mlir.py -h` for more usage info.

SRCS := $(wildcard circuit_qasm/*.qasm)
MLIR_QASM_SRCS := $(patsubst circuit_qasm/%.qasm, circuit_mlir/%.qasm.mlir, $(SRCS))
MLIR_QASM_OPT_SRCS := $(patsubst circuit_qasm/%.qasm, circuit_mlir/%.qasm.opt.mlir, $(SRCS))
MLIR_QSSA_SRCS := $(patsubst circuit_qasm/%.qasm, circuit_mlir/%.qssa.mlir, $(SRCS))
MLIR_QSSA_OPT_SRCS := $(patsubst circuit_qasm/%.qasm, circuit_mlir/%.qssa.opt.mlir, $(SRCS))
QASM_FINAL := $(patsubst circuit_qasm/%.qasm, circuit_mlir/%.final.qasm, $(SRCS))

PLOT_SRCS := $(wildcard plot-*.py)
PLOT_PDFS := $(patsubst %.py, %.pdf, $(PLOT_SRCS))

QUANTUM_OPT:=quantum-opt
QUANTUM_TRANSLATE:=quantum-translate
RESULTS_JSON:=gate-count-bench.json

all: $(MLIR_QASM_SRCS) $(MLIR_QASM_OPT_SRCS) $(MLIR_QSSA_SRCS) $(MLIR_QSSA_OPT_SRCS) $(QASM_FINAL)

qasm_mlir: $(MLIR_QASM_SRCS)
qssa_mlir: $(MLIR_QSSA_SRCS)
qssa_opt_mlir: $(MLIR_QSSA_OPT_SRCS)

# Pipeline:
# %.qasm -> %.qasm.mlir -> %.qasm.opt.mlir -> %.qssa.mlir > %.qssa.opt.mlir
#                                  |
#                                  v
#                              %.final.qasm

circuit_mlir/%.qasm.mlir: circuit_qasm/%.qasm
	python3 $(OPENQASM_TO_MLIR_PY) -i $< -o $@

circuit_mlir/%.qasm.opt.mlir: circuit_mlir/%.qasm.mlir
	$(QUANTUM_OPT) $^ --canonicalize --cse --symbol-dce --qasm-make-gates-opaque=gates=s,sdg,t,tdg,h,x,y,z,rx,ry,rz >$@

circuit_mlir/%.qssa.mlir: circuit_mlir/%.qasm.opt.mlir
	$(QUANTUM_OPT) $^ --convert-qasm-to-scf --convert-qasm-to-qssa >$@

circuit_mlir/%.qssa.opt.mlir: circuit_mlir/%.qssa.mlir
	$(QUANTUM_OPT) $^ --inline --cse --symbol-dce --qssa-apply-rewrites --qssa-convert-1q-to-U --qssa-apply-rewrites --canonicalize >$@

circuit_mlir/%.final.qasm: circuit_mlir/%.qasm.opt.mlir
	$(QUANTUM_TRANSLATE) --mlir-to-openqasm $^ >$@

convert_all:
	python3 $(OPENQASM_TO_MLIR_PY) --config convert.txt

# benchmarking & plotting
$(RESULTS_JSON): run-opts-bench.py $(MLIR_QSSA_SRCS)
	python3 run-opts-bench.py 

%.pdf: %.py $(RESULTS_JSON)
	python3 $< -i $(RESULTS_JSON)

plots: $(PLOT_PDFS)

clean:
	rm -f circuit_mlir/*.mlir circuit_mlir/*.qasm
	rm -f $(RESULTS_JSON) $(PLOT_PDFS)

.PHONY: convert_all clean plots

