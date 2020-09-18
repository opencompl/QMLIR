//===- quantum-simulator-wrappers.cpp - Quantum Dialect Simulation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper functions to execute operations from the quantum dialect.
// The quantum program will effectively be simulated using PRGs
// for measurements
//
//===----------------------------------------------------------------------===//

#include <array>
#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "memref-utils.h"
#include "simulator-utils.h"

using namespace std;

/// The main simulator module
unique_ptr<SimpleQuantumSimulator> simulator;

/// Quantum Simulator Library Functions
///  Functions exposed to MLIR code
///  Note: lowered `memref` arguments are unpacked and passed,
///        and have to be handled accordingly.

/// Initialize the simulator
extern "C" void __mlir_quantum_simulator__initialize(int64_t size) {
  // Currently no options available. Just initialize the simple simulator
  ::simulator = make_unique<SimpleQuantumSimulator>(size, time(NULL));
}

/// Allocates `size` qubits
extern "C" QubitSlice __mlir_quantum_simulator__acquire_qubits(int64_t size) {
  return simulator->acquireQubits(size);
}

/// Concats two qubit slices
extern "C" QubitSlice __mlir_quantum_simulator__concat_qubits(
    int64_t *allocatedPtr1, int64_t *alignedPtr1, int64_t offset1,
    array<int64_t, 1> sizes1, array<int64_t, 1> strides1,
    int64_t *allocatedPtr2, int64_t *alignedPtr2, int64_t offset2,
    array<int64_t, 1> sizes2, array<int64_t, 1> strides2) {
  QubitSlice arg1{allocatedPtr1, alignedPtr1, offset1, sizes1, strides1};
  QubitSlice arg2{allocatedPtr2, alignedPtr2, offset2, sizes2, strides2};
  return simulator->concatQubits(arg1, arg2);
}

/// Splits the qubit slice into two
extern "C" pair<QubitSlice, QubitSlice> __mlir_quantum_simulator__split_qubits(
    int64_t *allocatedPtr, int64_t *alignedPtr, int64_t offset,
    array<int64_t, 1> sizes, array<int64_t, 1> strides, int64_t size1,
    int64_t size2) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  return simulator->splitQubits(arg, size1, size2);
}

/// Measures the qubits and returns a boolean array
extern "C" ResultRef __mlir_quantum_simulator__measure_qubits(
    int64_t *allocatedPtr, int64_t *alignedPtr, int64_t offset,
    array<int64_t, 1> sizes, array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  return simulator->measureQubits(arg);
}

/// Gate Ops

/// Some useful constants
namespace SimulatorConstants {
const double sqrt2 = sqrt(2);
const double invSqrt2 = 1 / sqrt2;
const double pi = acos(-1);
const complex<double> i(0.0, 1.0);

const Matrix matH({{invSqrt2, invSqrt2}, {invSqrt2, -invSqrt2}});
const Matrix matX({{0, 1}, {1, 0}});
const Matrix matY({{0, -i}, {i, 0}});
const Matrix matZ({{1, 0}, {0, -1}});
}; // namespace SimulatorConstants

/// Hadamard Gate
extern "C" QubitSlice
__mlir_quantum_simulator__gate_H(int64_t *allocatedPtr, int64_t *alignedPtr,
                                 int64_t offset, array<int64_t, 1> sizes,
                                 array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  simulator->applyTransformToEach(arg, SimulatorConstants::matH);
  return arg;
}

extern "C" QubitSlice __mlir_quantum_simulator__gate_pauliX(
    int64_t *allocatedPtr, int64_t *alignedPtr, int64_t offset,
    array<int64_t, 1> sizes, array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  simulator->applyTransformToEach(arg, SimulatorConstants::matX);
  return arg;
}

extern "C" QubitSlice __mlir_quantum_simulator__gate_pauliY(
    int64_t *allocatedPtr, int64_t *alignedPtr, int64_t offset,
    array<int64_t, 1> sizes, array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  simulator->applyTransformToEach(arg, SimulatorConstants::matY);
  return arg;
}

extern "C" QubitSlice __mlir_quantum_simulator__gate_pauliZ(
    int64_t *allocatedPtr, int64_t *alignedPtr, int64_t offset,
    array<int64_t, 1> sizes, array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  simulator->applyTransformToEach(arg, SimulatorConstants::matZ);
  return arg;
}

/// Shows the full underlying qubit state
extern "C" void quantum_show_full_state() { simulator->showFullState(); }

/// Shows the state of a subset of qubits, provided they aren't entangled with
/// the rest
extern "C" void quantum_show_partial_state(int64_t *allocatedPtr,
                                           int64_t *alignedPtr, int64_t offset,
                                           array<int64_t, 1> sizes,
                                           array<int64_t, 1> strides) {
  QubitSlice arg{allocatedPtr, alignedPtr, offset, sizes, strides};
  simulator->showPartialState(arg);
}
