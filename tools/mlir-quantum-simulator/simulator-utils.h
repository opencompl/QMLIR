#ifndef QUANTUM_SIMULATOR__SIMULATOR_UTILS_H_
#define QUANTUM_SIMULATOR__SIMULATOR_UTILS_H_

#include "llvm/ADT/Twine.h"
#include <complex>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "matrix-utils.h"
#include "memref-utils.h"

/// Simulator Error helpers
namespace SimulatorLoggingSeverity {
enum SimulatorLoggingSeverity {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  CRITICAL = 4
};
} // namespace SimulatorLoggingSeverity

void simulatorLog(int severity, const char *op, const llvm::Twine &err);

using QubitSlice = MemRef1D<int64_t>;
using ResultRef = MemRef1D<bool>;

/// Interface for quantum simulators
class QuantumSimulator {
  // Helper functions for internal simulation
  std::mt19937 generator;
  std::uniform_real_distribution<> equiprobableDistribution;

public:
  /// Epsilon, for floating comparisions
  static constexpr double eps = 1e-8;

  /// Get a random real between 0 and 1
  virtual double getRandomReal();

  /// Return true with probability `p` (0 <= p <= 1)
  virtual bool getTrueWithProbP(double p);

  /// Check if two states are equal (including phase)
  virtual bool checkStatesEqual(const Ket &a, const Ket &b);

public:
  QuantumSimulator(uint64_t seed);
  virtual ~QuantumSimulator() {}

  /// Allocate and return a slice of `size` qubits.
  /// Underlying simulator is free to assign any qubits,
  ///  as long as they aren't already in use by the program
  /// If there aren't enough free qubit, print an error message and quit
  virtual QubitSlice acquireQubits(int64_t size) = 0;

  /// Concatenate two slices of qubits
  virtual QubitSlice concatQubits(const QubitSlice &q1,
                                  const QubitSlice &q2) = 0;

  /// Split a slice of qubits into two,
  /// with `size1` qubits on the left, and `size2` on the right
  /// If the sizes don't match, print an error message and quit
  virtual std::pair<QubitSlice, QubitSlice>
  splitQubits(const QubitSlice &q, int64_t size1, int64_t size2) = 0;

  /// Measure the qubits in the slice,
  ///  and release them back to the simulator
  /// Return a memref to a boolean array, storing the results
  virtual ResultRef measureQubits(const QubitSlice &q) = 0;

  /// Apply a particular gate on a qubit slice.
  /// Gate is provided as a matrix.
  /// If the sizes don't match, print an error message and quit
  virtual void applyTransform(const QubitSlice &q, const Matrix &gate) = 0;

  /// Apply a single qubit gate to all qubits in the slice
  /// `gate` has to be a 2x2 matrix
  virtual void applyTransformToEach(const QubitSlice &q,
                                    const Matrix &gate) = 0;

  /// Show the full underlying state of the simulator
  virtual void showFullState() = 0;

  /// Show the state of the selected qubits, if they are separable
  /// Otherwise print an error message, and continue silently
  virtual void showPartialState(const QubitSlice &q) = 0;
};

/// Qubit Register
/// Stores the entire register of qubits, as a single state vector
/// Each qubit array in the language corresponds to a subset of qubits in the
/// register
class QubitRegister {
  /// Number of qubits in the register
  int64_t numQubits;

  /// Dimension of the space (= size of the state vector)
  // CONSTRAINT dimension == 1 << numQubits
  int64_t dimension;

  /// The full state vector
  Ket state;

  /// A set of unused qubit indices, which can be allocated
  std::set<int64_t> unusedIndices;

  /// Permute the register. Place `i` at `perm[i]`
  void applyIndexPermutation(const std::vector<int64_t> &perm,
                             bool invert = false);

  /// Moves the subset of qubits to the end of the register
  void moveSliceToMSB(const std::vector<int64_t> &subset, bool invert = false);

  /// Moves the subset of qubits to the start of the register
  void moveSliceToLSB(const std::vector<int64_t> &subset, bool invert = false);

public:
  QubitRegister(int64_t num);

  QubitSlice acquireQubits(int64_t size);

  QubitSlice concatQubits(const QubitSlice &q1, const QubitSlice &q2);

  std::pair<QubitSlice, QubitSlice> splitQubits(const QubitSlice &q,
                                                int64_t size1, int64_t size2);

  ResultRef measureQubits(const QubitSlice &q, QuantumSimulator &simulator);

  void applyTransform(const QubitSlice &q, const Matrix &gate);

  void showFullState() const;
  void showPartialState(const QubitSlice &qs, QuantumSimulator &simulator);
};

/// Implements a single linear register simulator
class SimpleQuantumSimulator : public QuantumSimulator {
  std::unique_ptr<QubitRegister> qubitRegister;

public:
  SimpleQuantumSimulator(int64_t numQubits, uint64_t seed);
  ~SimpleQuantumSimulator();

  QubitSlice acquireQubits(int64_t size) override;
  QubitSlice concatQubits(const QubitSlice &q1, const QubitSlice &q2) override;
  std::pair<QubitSlice, QubitSlice>
  splitQubits(const QubitSlice &q, int64_t size1, int64_t size2) override;
  ResultRef measureQubits(const QubitSlice &q) override;

  void applyTransform(const QubitSlice &q, const Matrix &gate) override;
  void applyTransformToEach(const QubitSlice &q, const Matrix &gate) override;

  void showFullState() override;
  void showPartialState(const QubitSlice &q) override;
};

#endif // QUANTUM_SIMULATOR__SIMULATOR_UTILS_H_
