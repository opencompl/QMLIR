all: plot-gate-counts.pdf plot-times.pdf

plot-gate-counts.pdf: ./plot-gate-counts.py ./qasmbench-gate-count.json
	python3 $^
plot-times.pdf: ./plot-times.py ./qasmbench-gate-count.json
	python3 $^

