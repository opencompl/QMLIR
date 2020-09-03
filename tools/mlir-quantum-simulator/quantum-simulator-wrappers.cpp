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

#include <iostream>
#include <vector>
using namespace std;

using QubitSlice = vector<int>;

class QubitRegister {
  unsigned numQubits;
  unsigned long long size;
  vector<double> state;
  vector<int> unused;
public:
  QubitRegister(int numQubits) :
    numQubits(numQubits),
    size(1ll << numQubits),
    state(size, 0.0) {
    state[0] = 1.0;
  }

  // measures and releases qubits
  vector<bool> measure(const QubitSlice& idx) {
    return {};
  }
};


extern "C" void* acquire_qubits(int size) {
  return nullptr;
}

extern "C" int* measure(void* qs) {
  return nullptr;
}

extern "C" void printLn(int n) {
  cout << n << endl;
}
