#ifndef QUANTUM_SIMULATOR_MEMREF_UTILS_H_
#define QUANTUM_SIMULATOR_MEMREF_UTILS_H_

#include <array>
#include <vector>

/// MemRef Helpers
/// Used for allocating and managing Qubits
template <typename Elem>
struct MemRef1D {
  Elem *allocatedPtr;
  Elem *alignedPtr;
  int64_t offset;
  std::array<int64_t, 1> sizes;   // omitted when rank == 0
  std::array<int64_t, 1> strides; // omitted when rank == 0

  template <typename Index>
  Elem &operator[](const Index i);

  template <typename Index>
  const Elem &operator[](const Index i) const;

  /// return the size of the first (and only) dimension
  int64_t size() const;

  /// return the pointer to the first element
  Elem *begin() const;

  /// return the pointer to one past the last element
  Elem *end() const;

  /// Get a dynamically allocated array with `size` elements
  static MemRef1D<Elem> get(int64_t size);

  /// Get a dynamically allocated array with `size` elements
  ///  and initialize them with the values in `init`
  static MemRef1D<Elem> get(const std::vector<Elem> &init);

  void release() const;
};

template <typename Elem>
template <typename Index>
Elem &MemRef1D<Elem>::operator[](const Index i) {
  return alignedPtr[offset + strides[0] * i];
}

template <typename Elem>
template <typename Index>
const Elem &MemRef1D<Elem>::operator[](const Index i) const {
  return alignedPtr[offset + strides[0] * i];
}

template <typename Elem>
int64_t MemRef1D<Elem>::size() const {
  return sizes[0];
}

template <typename Elem>
Elem *MemRef1D<Elem>::begin() const {
  return alignedPtr;
}

template <typename Elem>
Elem *MemRef1D<Elem>::end() const {
  return alignedPtr + size();
}

template <typename Elem>
MemRef1D<Elem> MemRef1D<Elem>::get(int64_t size) {
  Elem *mem = new Elem[size];
  return MemRef1D<Elem>{mem, mem, 0, {size}, {1}};
}

template <typename Elem>
MemRef1D<Elem> MemRef1D<Elem>::get(const std::vector<Elem> &init) {
  auto res = MemRef1D<Elem>::get(init.size());
  std::copy(init.begin(), init.end(), res.alignedPtr);
  return res;
}

template <typename Elem>
void MemRef1D<Elem>::release() const {
  delete[] allocatedPtr;
}

#endif // QUANTUM_SIMULATOR_MEMREF_UTILS_H_
