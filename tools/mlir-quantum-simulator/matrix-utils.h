#ifndef QUANTUM_SIMULATOR__MATRIX_UTILS_H_
#define QUANTUM_SIMULATOR__MATRIX_UTILS_H_

#include <complex>
#include <vector>

using Complex = std::complex<double>;
using Ket = std::vector<Complex>;

/// Matrix helper class
class Matrix {
  /// Row-major storage as a superket
  int64_t numRows, numCols;
  Ket elems;

  /// Apply on range [st, en)
  void apply(Ket::iterator st, Ket::iterator en) const;

public:
  /// Create an empty Matrix
  Matrix();

  /// Create a matrix of dimensions `rows` x `cols`
  Matrix(int64_t rows, int64_t cols);

  /// Create a square matrix of dimensions `n` x `n`
  Matrix(int64_t n);

  /// Create a matrix from a 2D vector
  Matrix(const std::vector<Ket> &v);

  /// Accessors
  Complex at(int64_t i, int64_t j) const;
  Complex &at(int64_t i, int64_t j);

  /// Attributes
  int64_t getNumRows() const;
  int64_t getNumCols() const;

  /// right-multiply `other` and return the result
  Matrix multiply(const Matrix &other) const;

  /// Compute the tensor product of this and other
  Matrix tensorProduct(const Matrix &other) const;

  /// Apply on a state-ket
  void applyFull(Ket &ket) const;

  /// Apply partially on a key, starting at index `offset`
  void applyPartial(Ket &ket, int64_t offset = 0) const;

  /// Apply in blocks
  void applyBlocks(Ket &ket, int64_t offset = 0) const;
};

#endif // QUANTUM_SIMULATOR__MATRIX_UTILS_H_