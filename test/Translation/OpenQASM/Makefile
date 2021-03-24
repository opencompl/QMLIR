SRCS :=\
	E2-teleportation.qasm\
	E3-qft.qasm\
	E4-inverse-qft-v1.qasm\
	E4-inverse-qft-v2.qasm\
	E5-ripple-carry-adder.qasm\
	E6-randomized-bench.qasm\
	E7-quantum-process-tomography.qasm\
	E8-error-correction.qasm
MLIRS := $(patsubst %.qasm, %.qasm.mlir, $(SRCS))
OPT_MLIRS := $(patsubst %.qasm, %.qasm.opt.mlir, $(SRCS))
FINAL := $(patsubst %.qasm, %.final.qasm, $(SRCS))

all: $(FINAL)

# Pipeline: %.qasm -> %.qasm.mlir -> %.qasm.opt.mlir -> %.final.qasm
%.qasm.mlir: %.qasm
	python ../../../tools/openqasm-to-mlir.py -i $< -o $@

%.qasm.opt.mlir: %.qasm.mlir
	quantum-opt $^ | quantum-opt --cse --canonicalize --cse --symbol-dce >$@

%.final.qasm: %.qasm.opt.mlir
	quantum-translate --mlir-to-openqasm $^ >$@

clean:
	rm $(MLIRS) $(OPT_MLIRS) $(FINAL) -f

.PHONY: clean

