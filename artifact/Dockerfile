FROM ubuntu:20.04
WORKDIR /artifact

### base
RUN apt update -y
ARG DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND="noninteractive" apt install -y tzdata
RUN apt install -y 
RUN apt install -y build-essential
RUN apt install -y cmake
RUN apt install -y git
RUN apt install -y ninja-build
RUN apt install -y python3.8 python3-pip
RUN apt install -y lld
RUN apt install -y vim
RUN apt install -y wget curl
RUN apt install -y lsb-release software-properties-common
RUN apt install -y zip unzip
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

### project dependencies
# 1. LLVM
RUN git clone https://github.com/llvm/llvm-project
RUN cd /artifact/llvm-project && git checkout 9816d43cff5ad7abb94eab94dcd205285675972a
RUN cd /artifact/llvm-project && \
    mkdir build &&                \
    cd build &&                    \
    cmake ../llvm -G Ninja          \
      -DCMAKE_BUILD_TYPE=Release     \
      -DCMAKE_C_COMPILER=clang-13     \
      -DCMAKE_CXX_COMPILER=clang++-13  \
      -DLLVM_ENABLE_ASSERTIONS=ON       \
      -DLLVM_BUILD_EXAMPLES=OFF          \
      -DLLVM_TARGETS_TO_BUILD="host"      \
      -DLLVM_ENABLE_PROJECTS='mlir'        \
      -DLLVM_ENABLE_OCAMLDOC=OFF            \
      -DLLVM_ENABLE_BINDINGS=OFF             \
      -DLLVM_INSTALL_UTILS=ON                 \
      -DLLVM_ENABLE_LLD=ON &&                  \
    ninja check-mlir

# 2. Python packages
RUN pip3 install matplotlib
RUN pip3 install qiskit
RUN pip3 install numpy
RUN pip3 install sh

### Project
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

# Benchmarks
# 1. QASMBench
RUN git clone https://github.com/anurudhp/QASMBench.git -b cc-artifact
RUN cd /artifact/QASMBench && \
    mkdir circuit_qasm && \
    cp small/*/*.qasm medium/*/*.qasm large/*/*.qasm circuit_qasm/
RUN cp /artifact/QMLIR/data/scripts/Makefile \
       /artifact/QMLIR/data/scripts/run-opts-bench.py \
       /artifact/QASMBench/
RUN cd /artifact/QASMBench && \
    mkdir circuit_mlir && \
    make all
RUN cd /artifact/QASMBench && \
    python3 run-opts-bench.py -o qasmbench-results.json

# 2. IBM Challenge
RUN mkdir /artifact/IBMChallenge && \
    cd /artifact/IBMChallenge && \
    cp /artifact/QMLIR/data/circuit_qasm.zip . && \
    unzip circuit_qasm.zip
RUN cp /artifact/QMLIR/data/scripts/Makefile \
       /artifact/QMLIR/data/scripts/run-opts-bench.py \
       /artifact/IBMChallenge/
RUN cd /artifact/IBMChallenge && \
    mkdir circuit_mlir && \
    make all
RUN cd /artifact/IBMChallenge && \
    python3 run-opts-bench.py -o ibmchallenge-results.json
