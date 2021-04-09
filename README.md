# An attempt at making a Quantum Dialect for MLIR

This repository uses a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a standalone `opt`-like tool to operate on that dialect.   
Template taken from [jmgorius/mlir-standalone-template](https://github.com/jmgorius/mlir-standalone-template)

## How to build

Clone all submodules:
```sh
git submodule init
git submodule update
```

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-quantum-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## LLVM Build Info

Check the [LLVM COMMIT INFO](https://github.com/anurudhp/QMLIR/blob/master/.github/workflows/build-and-test.yml#L6) for latest verified build.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
