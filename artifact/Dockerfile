FROM ubuntu:20.04
WORKDIR /artifact

### base
RUN apt-get update -y
ARG DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y ninja-build
RUN apt-get install -y python3.8 python3-pip
RUN apt-get install -y lld
RUN apt-get install -y vim
RUN apt-get install -y wget curl
RUN apt-get install -y lsb-release software-properties-common
RUN apt-get install -y zip unzip
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && ./llvm.sh 13 && rm llvm.sh

### project dependencies
# 1. LLVM
RUN git clone https://github.com/llvm/llvm-project
RUN cd /artifact/llvm-project && git checkout 9816d43cff5ad7abb94eab94dcd205285675972a
RUN cd /artifact/llvm-project && \
    mkdir build && \
    cd build && \
    cmake ../llvm -G Ninja         \
      -DCMAKE_BUILD_TYPE=Release    \
      -DCMAKE_C_COMPILER=clang-13    \
      -DCMAKE_CXX_COMPILER=clang++-13 \
      -DLLVM_ENABLE_ASSERTIONS=ON      \
      -DLLVM_BUILD_EXAMPLES=OFF         \
      -DLLVM_TARGETS_TO_BUILD="host"     \
      -DLLVM_ENABLE_PROJECTS='mlir'       \
      -DLLVM_ENABLE_OCAMLDOC=OFF           \
      -DLLVM_ENABLE_BINDINGS=OFF            \
      -DLLVM_INSTALL_UTILS=ON                \
      -DLLVM_ENABLE_LLD=ON
RUN cd /artifact/llvm-project/build && ninja check-mlir

# 2. Python packages
RUN pip3 install matplotlib
RUN pip3 install qiskit
RUN pip3 install numpy
RUN pip3 install sh

### Project
RUN echo "bump-git-11"
RUN git clone https://github.com/opencompl/QMLIR
RUN cd /artifact/QMLIR && git checkout cc-artifact
RUN cd /artifact/QMLIR && git submodule init && git submodule update
RUN cd /artifact/QMLIR && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja                                          \
      -DCMAKE_BUILD_TYPE=Release                                \
      -DCMAKE_LINKER=lld                                         \
      -DCMAKE_C_COMPILER=clang-13                                 \
      -DCMAKE_CXX_COMPILER=clang++-13                              \
      -DMLIR_DIR=/artifact/llvm-project/build/lib/cmake/mlir/       \
      -DLLVM_DIR=/artifact/llvm-project/build/lib/cmake/llvm/        \
      -DLLVM_EXTERNAL_LIT=/artifact/llvm-project/build/bin/llvm-lit &&\
    ninja && \
    ninja check-quantum-opt
ENV PATH $PATH:/artifact/llvm-project/build/bin:/artifact/QMLIR/build/bin
ENV OPENQASM_TO_MLIR_PY /artifact/QMLIR/tools/openqasm-to-mlir.py

### Benchmarks
# 1. QASMBench
RUN git clone https://github.com/anurudhp/QASMBench.git -b cc-artifact
RUN cd /artifact/QASMBench && \
    mkdir circuit_qasm circuit_mlir && \
    cp small/*/*.qasm medium/*/*.qasm large/*/*.qasm circuit_qasm/
RUN cp /artifact/QMLIR/data/scripts/Makefile \
       /artifact/QMLIR/data/scripts/run-opts-bench.py \
       /artifact/QMLIR/data/scripts/plot-dataset-gate-stats.py \
       /artifact/QMLIR/data/scripts/plot-bench-gate-stats.py \
       /artifact/QMLIR/data/scripts/plot-bench-runtimes.py \
       /artifact/QASMBench/

# 2. IBM Challenge
RUN mkdir /artifact/IBMChallenge && \
    cd /artifact/IBMChallenge && \
    cp /artifact/QMLIR/data/circuit_qasm.zip . && \
    unzip circuit_qasm.zip && \
    mkdir circuit_mlir
RUN cp /artifact/QMLIR/data/scripts/Makefile \
       /artifact/QMLIR/data/scripts/run-opts-bench.py \
       /artifact/QMLIR/data/scripts/plot-dataset-gate-stats.py \
       /artifact/QMLIR/data/scripts/plot-bench-gate-stats.py \
       /artifact/QMLIR/data/scripts/plot-bench-runtimes.py \
       /artifact/QMLIR/data/scripts/plot-bench-runtimes-ibmchallenge.py \
       /artifact/IBMChallenge/

### Execute and plot
RUN cd /artifact/QASMBench && make plots
RUN cd /artifact/IBMChallenge && make plots
