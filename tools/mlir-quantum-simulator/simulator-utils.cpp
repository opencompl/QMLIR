#include <iostream>

#include "simulator-utils.h"

using namespace std;
using namespace llvm;

void simulatorLog(int severity, const char *op, const Twine &err) {
  switch (severity) {
  case SimulatorLoggingSeverity::DEBUG:
    cerr << "[DEBUG]";
    break;
  case SimulatorLoggingSeverity::INFO:
    cerr << "[INFO]";
    break;
  case SimulatorLoggingSeverity::WARN:
    cerr << "[WARN]";
    break;
  case SimulatorLoggingSeverity::ERROR:
    cerr << "[ERROR]";
    break;
  case SimulatorLoggingSeverity::CRITICAL:
    cerr << "[CRITICAL]";
    break;
  }

  cerr << " " << op << " - " << err.str() << endl;

  if (severity >= SimulatorLoggingSeverity::ERROR)
    exit(0);
}

//============================================================================//
// QubitRegister
//============================================================================//

QubitRegister::QubitRegister(int64_t num)
    : numQubits(num), dimension(1ll << num), state(dimension, 0.0),
      unusedIndices() {
  // initial state is all 0s
  state[0] = 1.0;

  // All qubits are free at the start
  for (int64_t i = 0; i < num; i++) {
    unusedIndices.insert(i);
  }
}

void QubitRegister::applyIndexPermutation(const vector<int64_t> &perm,
                                          bool invert) {
  assert((int64_t)perm.size() == numQubits);

  Ket stateCopy(state.size());

  for (int64_t mask = 0; mask < (1ll << numQubits); mask++) {
    int64_t newMask = 0;
    for (int64_t i = 0; i < numQubits; i++) {
      if (!invert) {
        // i -> perm[i]
        if (mask & (1ll << i)) {
          newMask |= 1ll << perm[i];
        }
      } else {
        // perm[i] -> i
        if (mask & (1ll << perm[i])) {
          newMask |= 1ll << i;
        }
      }
    }
    stateCopy[newMask] = state[mask];
  }

  swap(state, stateCopy);
}

void QubitRegister::moveSliceToMSB(const vector<int64_t> &subset, bool invert) {
  vector<int64_t> perm(numQubits, -1);
  int64_t idx = numQubits - 1;
  for (auto i : subset) {
    perm[i] = idx--;
  }
  for (auto &i : perm) {
    if (i == -1)
      i = idx--;
  }

  applyIndexPermutation(perm, invert);
}

void QubitRegister::moveSliceToLSB(const vector<int64_t> &subset, bool invert) {
  vector<int64_t> perm(numQubits, -1);
  int64_t idx = 0;
  for (auto i : subset) {
    perm[i] = idx++;
  }
  for (auto &i : perm) {
    if (i == -1)
      i = idx++;
  }

  applyIndexPermutation(perm, invert);
}

QubitSlice QubitRegister::acquireQubits(int64_t size) {
  if ((int64_t)unusedIndices.size() < size) {
    simulatorLog(SimulatorLoggingSeverity::ERROR, "allocate",
                 Twine("Unable to allocate ")
                     .concat(Twine(size))
                     .concat(" qubits, only have ")
                     .concat(Twine(unusedIndices.size()))
                     .concat(" qubits left"));
  }

  vector<int64_t> allocated;
  for (auto i : unusedIndices) {
    allocated.push_back(i);
    if ((int64_t)allocated.size() == size)
      break;
  }
  for (int i : allocated) {
    unusedIndices.erase(i);
  }
  return QubitSlice::get(allocated);
}

QubitSlice QubitRegister::concatQubits(const QubitSlice &q1,
                                       const QubitSlice &q2) {
  vector<int64_t> combined;

  combined.insert(combined.end(), q1.begin(), q1.end());
  q1.release();

  combined.insert(combined.end(), q2.begin(), q2.end());
  q2.release();

  return QubitSlice::get(combined);
}

pair<QubitSlice, QubitSlice>
QubitRegister::splitQubits(const QubitSlice &q, int64_t size1, int64_t size2) {
  if (q.size() != size1 + size2) {
    simulatorLog(SimulatorLoggingSeverity::ERROR, "split",
                 Twine("Mismatched split sizes, got ")
                     .concat(Twine(q.size()))
                     .concat(" qubits, but requested to split into ")
                     .concat(Twine(size1))
                     .concat(" + ")
                     .concat(Twine(size2)));
  }

  QubitSlice q1 =
      QubitSlice::get(vector<int64_t>(q.begin(), q.begin() + size1));
  QubitSlice q2 = QubitSlice::get(vector<int64_t>(q.begin() + size1, q.end()));
  q.release();

  return {q1, q2};
}

ResultRef QubitRegister::measureQubits(const QubitSlice &q,
                                       QuantumSimulator &simulator) {
  ResultRef result = ResultRef::get(q.size());

  // Moving to MSB makes measurement simple
  for (int64_t index = 0; index < q.size(); index++) {
    moveSliceToMSB({q[index]});

    double p0 = 0;
    for (int64_t i = 0; i < (int64_t)state.size() / 2; i++) {
      p0 += abs(state[i] * conj(state[i]));
    }
    double p1 = 1.0 - p0;

    if (simulator.getTrueWithProbP(p0)) {
      double scale = sqrt(p0);
      // measured |0⟩
      // scale the first half
      for (auto it = state.begin(); it != state.begin() + state.size() / 2;
           it++) {
        *it /= scale;
      }
      // set the second half to 0
      fill(state.begin() + state.size() / 2, state.end(), 0.0);

      result[index] = 0;
    } else {
      double scale = sqrt(p1);
      // measure |1⟩
      // set the first half to zero
      fill(state.begin(), state.begin() + state.size() / 2, 0.0);
      // scale the second half
      for (auto it = state.begin() + state.size() / 2; it != state.end();
           it++) {
        *it /= scale;
      }

      result[index] = 1;
    }

    moveSliceToMSB({q[index]}, true);
    unusedIndices.insert(q[index]);
  }

  // release the allocated qubit slice memory
  q.release();

  return result;
}

void QubitRegister::applyTransform(const QubitSlice &q, const Matrix &gate) {
  vector<int64_t> subset(q.begin(), q.end());

  // move current qubits to the front
  moveSliceToLSB(subset);

  // apply the operation, in blocks
  // as the qubits are at LSB, contiguous blocks are fixed states for the
  // rest of the qubits
  gate.applyBlocks(state);

  // undo the move
  moveSliceToLSB(subset, true);
}

void QubitRegister::showFullState() const {
  cerr << "> state = {";
  llvm::interleave(
      state,
      [&](const complex<double> &v) {
        cerr << v.real() << "+" << v.imag() << "i";
      },
      [&]() { cerr << ", "; });
  cerr << "}" << endl;
}

void QubitRegister::showPartialState(const QubitSlice &qs,
                                     QuantumSimulator &simulator) {
  // Equivalent to measuring the remaining qubits
  // If the state is dependent on the measurement outcome, then the qubits are
  // entangled. In that case, display an error message.

  vector<int64_t> usedQubits(qs.begin(), qs.end());
  llvm::sort(usedQubits);
  vector<int64_t> remainingQubits;
  for (int64_t i = 0, j = 0, k = 0; i < this->numQubits; i++) {
    if (j < (int64_t)usedQubits.size() && usedQubits[j] == i) {
      j++;
    } else {
      remainingQubits.push_back(i);
      k++;
    }
  }

  this->moveSliceToMSB(remainingQubits);

  const int64_t usedDimension = 1ll << usedQubits.size(),
                remainingDimension = 1ll << remainingQubits.size();
  llvm::Optional<Ket> partialState;
  for (int64_t base = 0; base < remainingDimension; base++) {
    Ket current(state.begin() + base * usedDimension,
                state.begin() + (base + 1) * usedDimension);

    double p2 =
        std::accumulate(current.begin(), current.end(), 0.0,
                        [&](double d, const complex<double> &a) -> double {
                          return d + abs(a * conj(a));
                        });

    if (p2 < QuantumSimulator::eps)
      continue;

    if (!partialState)
      partialState = current;
    if (!simulator.checkStatesEqual(partialState.getValue(), current)) {
      partialState.reset();
      break;
    }
  }

  this->moveSliceToMSB(remainingQubits, /*invert=*/true);

  if (!partialState) {
    cerr << "<error printing partial state: qubits entangled>\n";
    return;
  }

  cerr << "> state[";
  llvm::interleave(
      vector<int64_t>(qs.begin(), qs.end()), [&](int64_t x) { cerr << x; },
      [&]() { cerr << ", "; });
  cerr << "] = {";
  llvm::interleave(
      partialState.getValue(),
      [&](const complex<double> &v) {
        cerr << v.real() << "+" << v.imag() << "i";
      },
      [&]() { cerr << ", "; });
  cerr << "}" << endl;
}

//============================================================================//
// QuantumSimulator
//============================================================================//
QuantumSimulator::QuantumSimulator(uint64_t seed)
    : generator(seed), equiprobableDistribution(0.0, 1.0) {}

double QuantumSimulator::getRandomReal() {
  return equiprobableDistribution(generator);
}

bool QuantumSimulator::getTrueWithProbP(double p) {
  return getRandomReal() <= p;
}
bool QuantumSimulator::checkStatesEqual(const Ket &a, const Ket &b) {
  if (a.size() != b.size())
    return false;

  for (size_t i = 0; i < a.size(); i++) {
    auto diff = a[i] - b[i];
    if (abs(diff) > this->eps)
      return false;
  }

  return true;
}

//============================================================================//
// SimpleQuantumSimulator
//============================================================================//

SimpleQuantumSimulator::SimpleQuantumSimulator(int64_t numQubits, uint64_t seed)
    : QuantumSimulator(seed) {
  qubitRegister = make_unique<QubitRegister>(numQubits);
}
SimpleQuantumSimulator::~SimpleQuantumSimulator() {}

// Simulation support functions
QubitSlice SimpleQuantumSimulator::acquireQubits(int64_t size) {
  return qubitRegister->acquireQubits(size);
}
QubitSlice SimpleQuantumSimulator::concatQubits(const QubitSlice &q1,
                                                const QubitSlice &q2) {
  return qubitRegister->concatQubits(q1, q2);
}
pair<QubitSlice, QubitSlice>
SimpleQuantumSimulator::splitQubits(const QubitSlice &q, int64_t size1,
                                    int64_t size2) {
  return qubitRegister->splitQubits(q, size1, size2);
}
ResultRef SimpleQuantumSimulator::measureQubits(const QubitSlice &q) {
  return qubitRegister->measureQubits(q, *this);
}

void SimpleQuantumSimulator::showFullState() { qubitRegister->showFullState(); }

void SimpleQuantumSimulator::showPartialState(const QubitSlice &qs) {
  qubitRegister->showPartialState(qs, *this);
}

void SimpleQuantumSimulator::applyTransform(const QubitSlice &q,
                                            const Matrix &gate) {
  qubitRegister->applyTransform(q, gate);
}

void SimpleQuantumSimulator::applyTransformToEach(const QubitSlice &q,
                                                  const Matrix &gate) {
  if (q.size() == 0)
    return;
  Matrix fullGate = gate;
  for (int64_t i = 1; i < q.size(); i++) {
    fullGate = fullGate.tensorProduct(gate);
  }
  applyTransform(q, fullGate);
}
