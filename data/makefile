.PHONY: all

all: plot-gate-count.pdf

gate-count-data.json:
	./gen-gate-count.py

plot-gate-count.pdf: gate-count-data.json
	./plot-gate-count.py
