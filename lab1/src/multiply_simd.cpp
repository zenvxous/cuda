#include <cstring>
#include <emmintrin.h>

#include "block_matrix.hpp"
#include "multiply.hpp"

namespace lab1 {
namespace {

constexpr int kVectors = kBlk / 2;
constexpr int kTail = kBlk % 2;
constexpr int kAccs = kVectors > 0 ? kVectors : 1;
constexpr bool kAligned = (kBlk % 2 == 0);
constexpr int kXmmCount = 16;
constexpr int kRowStep = (2 * kVectors + 3 <= kXmmCount - 3) ? 2 : 1;

inline __m128d load_row(const Elem* p) {
  if constexpr (kAligned) return _mm_load_pd(p);
  else return _mm_loadu_pd(p);
}

inline void store_row(Elem* p, __m128d v) {
  if constexpr (kAligned) _mm_store_pd(p, v);
  else _mm_storeu_pd(p, v);
}

template <int Rows>
inline void accumulate(const Elem* a_block, const Elem* b_block, Elem* c_block, int r) {
  __m128d acc[Rows][kAccs];
  [[maybe_unused]] Elem tail[Rows];

  for (int q = 0; q < Rows; ++q) {
    const Elem* const c_row = c_block + (r + q) * kBlk;
    for (int v = 0; v < kVectors; ++v) acc[q][v] = load_row(c_row + 2 * v);
    if constexpr (kTail != 0) tail[q] = c_row[kBlk - 1];
  }

  for (int k = 0; k < kBlk; ++k) {
    const Elem* const b_row = b_block + k * kBlk;
    __m128d a_splat[Rows];
    for (int q = 0; q < Rows; ++q) a_splat[q] = _mm_set1_pd(a_block[(r + q) * kBlk + k]);

    for (int v = 0; v < kVectors; ++v) {
      const __m128d b_vec = load_row(b_row + 2 * v);
      for (int q = 0; q < Rows; ++q) {
        acc[q][v] = _mm_add_pd(acc[q][v], _mm_mul_pd(a_splat[q], b_vec));
      }
    }
    if constexpr (kTail != 0) {
      for (int q = 0; q < Rows; ++q) {
        tail[q] += a_block[(r + q) * kBlk + k] * b_row[kBlk - 1];
      }
    }
  }

  for (int q = 0; q < Rows; ++q) {
    Elem* const c_row = c_block + (r + q) * kBlk;
    for (int v = 0; v < kVectors; ++v) store_row(c_row + 2 * v, acc[q][v]);
    if constexpr (kTail != 0) c_row[kBlk - 1] = tail[q];
  }
}

}

void multiply_simd(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c) {
  const int block_rows = a.rows();
  const int block_inner = a.cols();
  const int block_cols = b.cols();

  std::memset(c.data(), 0, c.bytes());

  for (int i = 0; i < block_rows; ++i) {
    for (int k_block = 0; k_block < block_inner; ++k_block) {
      const Elem* const __restrict a_block = a.block(i, k_block);

      for (int j = 0; j < block_cols; ++j) {
        const Elem* const __restrict b_block = b.block(k_block, j);
        Elem* const __restrict c_block = c.block(i, j);

        int r = 0;
        for (; r + kRowStep <= kBlk; r += kRowStep) {
          accumulate<kRowStep>(a_block, b_block, c_block, r);
        }
        for (; r < kBlk; ++r) accumulate<1>(a_block, b_block, c_block, r);
      }
    }
  }
}

}
