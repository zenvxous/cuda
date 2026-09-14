#include "block_matrix.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

namespace lab1 {
namespace {

constexpr std::size_t kAlignment = 64;

inline std::uint64_t next_random(std::uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}

}

BlockMatrix::BlockMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
  void* raw = nullptr;
  if (bytes() != 0 && posix_memalign(&raw, kAlignment, bytes()) != 0) throw std::bad_alloc();
  data_ = static_cast<Elem*>(raw);
}

BlockMatrix::~BlockMatrix() { std::free(data_); }

void fill_random(BlockMatrix& m, std::uint64_t seed) {
  std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
  Elem* const p = m.data();
  const std::size_t n = m.elem_count();
  for (std::size_t i = 0; i < n; ++i) {
    const double unit = static_cast<double>(next_random(state) >> 11) / 9007199254740992.0;
    p[i] = 2.0 * unit - 1.0;
  }
}

void fill_zero(BlockMatrix& m) {
  if (m.data() != nullptr) std::memset(m.data(), 0, m.bytes());
}

CompareResult compare(const BlockMatrix& a, const BlockMatrix& b) {
  CompareResult result;
  if (a.rows() != b.rows() || a.cols() != b.cols()) return result;

  result.bit_exact = std::memcmp(a.data(), b.data(), a.bytes()) == 0;

  const Elem* pa = a.data();
  const Elem* pb = b.data();
  const std::size_t n = a.elem_count();
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = std::fabs(pa[i] - pb[i]);
    if (diff != 0.0) {
      ++result.diff_count;
      if (diff > result.max_abs_diff) result.max_abs_diff = diff;
    }
  }
  return result;
}

double checksum(const BlockMatrix& m) {
  const Elem* p = m.data();
  const std::size_t n = m.elem_count();
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += p[i];
  return sum;
}

}
