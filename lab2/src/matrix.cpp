#include "matrix.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

namespace lab2 {
namespace {

inline std::uint64_t next_random(std::uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}

}

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
  stride_ = ((cols + kStrideAlignElems - 1) / kStrideAlignElems) * kStrideAlignElems;
  void* raw = nullptr;
  if (bytes() != 0 && posix_memalign(&raw, kAlignment, bytes()) != 0) throw std::bad_alloc();
  data_ = static_cast<Elem*>(raw);
  if (data_ != nullptr) std::memset(data_, 0, bytes());
}

Matrix::~Matrix() { std::free(data_); }

void fill_random(Matrix& m, std::uint64_t seed) {
  std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
  for (int i = 0; i < m.rows(); ++i) {
    Elem* const r = m.row(i);
    for (int j = 0; j < m.cols(); ++j) {
      const double unit = static_cast<double>(next_random(state) >> 11) / 9007199254740992.0;
      r[j] = 2.0 * unit - 1.0;
    }
  }
}

void fill_zero(Matrix& m) {
  if (m.data() != nullptr) std::memset(m.data(), 0, m.bytes());
}

CompareResult compare(const Matrix& a, const Matrix& b) {
  CompareResult result;
  if (a.rows() != b.rows() || a.cols() != b.cols()) return result;

  result.bit_exact = true;
  for (int i = 0; i < a.rows(); ++i) {
    const Elem* const pa = a.row(i);
    const Elem* const pb = b.row(i);
    if (std::memcmp(pa, pb, static_cast<std::size_t>(a.cols()) * sizeof(Elem)) != 0) {
      result.bit_exact = false;
    }
    for (int j = 0; j < a.cols(); ++j) {
      const double diff = std::fabs(pa[j] - pb[j]);
      if (diff != 0.0) {
        ++result.diff_count;
        if (diff > result.max_abs_diff) result.max_abs_diff = diff;
      }
    }
  }
  return result;
}

double checksum(const Matrix& m) {
  double sum = 0.0;
  for (int i = 0; i < m.rows(); ++i) {
    const Elem* const r = m.row(i);
    for (int j = 0; j < m.cols(); ++j) sum += r[j];
  }
  return sum;
}

}
