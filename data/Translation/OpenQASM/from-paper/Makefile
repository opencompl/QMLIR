SRCS :=\
	E2-teleportation.qasm\
	E3-qft.qasm\
	E4-inverse-qft-v1.qasm\
	E4-inverse-qft-v2.qasm\
	E5-ripple-carry-adder.qasm\
	E6-randomized-bench.qasm\
	E7-quantum-process-tomography.qasm\
	E8-error-correction.qasm
MLIRS_1 := $(patsubst %.qasm, %.1.qasm.mlir, $(SRCS))
OPT_MLIRS_1 := $(patsubst %.qasm, %.1.qasm.opt.mlir, $(SRCS))
FINAL_QASM_1 := $(patsubst %.qasm, %.1.final.qasm, $(SRCS))
MLIRS_2 := $(patsubst %.qasm, %.2.qasm.mlir, $(SRCS))
OPT_MLIRS_2 := $(patsubst %.qasm, %.2.qasm.opt.mlir, $(SRCS))
FINAL_QASM_2 := $(patsubst %.qasm, %.2.final.qasm, $(SRCS))

all: roundtrip


# Pipeline:
# %.qasm
# -> %.1.qasm.mlir
# -> %.1.qasm.opt.mlir
# -> %.1.final.qasm
# -> %.2.qasm.mlir
# -> %.2.qasm.opt.mlir
# -> %.2.final.qasm

roundtrip: $(MLIRS_1) $(OPT_MLIRS_1) $(FINAL_QASM_1) $(MLIRS_2) $(OPT_MLIRS_2) $(FINAL_QASM_2)

%.1.qasm.mlir: %.qasm
	python ../../../tools/openqasm-to-mlir.py -i $< -o $@

%.2.qasm.mlir: %.1.final.qasm
	python ../../../tools/openqasm-to-mlir.py -i $< -o $@

%.qasm.opt.mlir: %.qasm.mlir
	quantum-opt $^ | quantum-opt --canonicalize --cse --inline --cse --symbol-dce >$@

%.final.qasm: %.qasm.opt.mlir
	quantum-translate --mlir-to-openqasm $^ >$@

clean:
	rm $(MLIRS_1) $(OPT_MLIRS_1) $(FINAL_QASM_1) $(MLIRS_2) $(OPT_MLIRS_2) $(FINAL_QASM_2) -f

.PHONY: clean roundtrip diffcheck

