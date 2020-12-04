#include <cassert>

#include "matrix-utils.h"

using namespace std;
// using namespace llvm;

Matrix::Matrix() : numRows(0), numCols(0), elems(0, 0.0) {}

Matrix::Matrix(int64_t rows, int64_t cols)
    : numRows(rows), numCols(cols), elems(rows * cols, 0.0) {}

Matrix::Matrix(int64_t n) : numRows(n), numCols(n), elems(n * n, 0.0) {}

Matrix::Matrix(const vector<Ket> &v) {
  if (v.empty()) {
    numRows = numCols = 0;
  } else {
    numRows = v.size();
    numCols = v[0].size();
    elems.resize(numRows * numCols, 0);
    for (int64_t i = 0; i < numRows; i++) {
      for (int64_t j = 0; j < numCols; j++) {
        at(i, j) = v[i][j];
      }
    }
  }
}

Complex Matrix::at(int64_t i, int64_t j) const {
  return elems[i * numCols + j];
}

Complex &Matrix::at(int64_t i, int64_t j) { return elems[i * numCols + j]; }

int64_t Matrix::getNumRows() const { return numRows; }

int64_t Matrix::getNumCols() const { return numCols; }

Matrix Matrix::multiply(const Matrix &other) const {
  assert(numCols == other.numRows && "invalid dimensions for multiplication");

  Matrix res(numRows, other.numCols);
  for (int64_t i = 0; i < res.getNumRows(); i++) {
    for (int64_t j = 0; j < res.getNumCols(); j++) {
      for (int64_t k = 0; k < numCols; k++) {
        res.at(i, j) += at(i, k) * other.at(k, j);
      }
    }
  }

  return res;
}

Matrix Matrix::tensorProduct(const Matrix &other) const {
  Matrix res(numRows * other.numRows, numCols * other.numCols);

  for (int64_t i = 0; i < numRows; i++) {
    for (int64_t ii = 0; ii < other.numRows; ii++) {
      for (int64_t j = 0; j < numCols; j++) {
        for (int64_t jj = 0; jj < other.numCols; jj++) {
          res.at(i * other.numRows + ii, j * other.numCols + jj) =
              at(i, j) * other.at(ii, jj);
        }
      }
    }
  }

  return res;
}

void Matrix::applyFull(Ket &ket) const { apply(ket.begin(), ket.end()); }

void Matrix::applyPartial(Ket &ket, int64_t offset) const {
  assert(offset + numCols <= (int64_t)ket.size() &&
         "Invalid offset, too few elements");
  apply(ket.begin() + offset, ket.begin() + offset + numCols);
}

void Matrix::applyBlocks(Ket &ket, int64_t offset) const {
  assert(offset >= 0);
  assert(offset < (int64_t)ket.size() && "Invalid offset, too few elements");

  int64_t numBlocks = (ket.size() - offset) / numCols;
  assert(numBlocks * numCols + offset == (int64_t)ket.size() &&
         "Invalid offset and/or remaining size, unable to segment");

  for (int64_t i = 0; i < numBlocks; i++) {
    auto st = ket.begin() + offset + i * numCols;
    apply(st, st + numCols);
  }
}

void Matrix::apply(Ket::iterator st, Ket::iterator en) const {
  assert(en - st == numCols && "mismatched Ket size and Matrix numCols");
  assert(numCols == numRows && "non-square matrices are unsupported");

  Ket temp(numRows, 0.0);
  for (int64_t i = 0; i < numRows; i++) {
    for (int64_t j = 0; j < numCols; j++) {
      temp[i] += at(i, j) * st[j];
    }
  }

  copy(temp.begin(), temp.end(), st);
}
