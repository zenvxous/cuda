#include <cstring>
#include <arm_neon.h>

#include "block_matrix.hpp"
#include "multiply.hpp"

#ifndef LAB1_ROW_STEP
#define LAB1_ROW_STEP 2
#endif

namespace lab1 {
namespace {

constexpr int kVectors = kBlk / 2;
constexpr int kTail = kBlk % 2;
constexpr int kAccs = kVectors > 0 ? kVectors : 1;
constexpr int kVecRegisters = 32;
constexpr int kRowStep = (LAB1_ROW_STEP * kVectors + LAB1_ROW_STEP + 1 <= kVecRegisters - 4)
                             ? LAB1_ROW_STEP
                             : 1;

template <int Rows>
inline void accumulate(const Elem* a_block, const Elem* b_block, Elem* c_block, int r) {
  float64x2_t acc[Rows][kAccs];
  [[maybe_unused]] Elem tail[Rows];

  for (int q = 0; q < Rows; ++q) {
    const Elem* const c_row = c_block + (r + q) * kBlk;
    for (int v = 0; v < kVectors; ++v) acc[q][v] = vld1q_f64(c_row + 2 * v);
    if constexpr (kTail != 0) tail[q] = c_row[kBlk - 1];
  }

  for (int k = 0; k < kBlk; ++k) {
    const Elem* const b_row = b_block + k * kBlk;
    float64x2_t a_splat[Rows];
    for (int q = 0; q < Rows; ++q) a_splat[q] = vdupq_n_f64(a_block[(r + q) * kBlk + k]);

    for (int v = 0; v < kVectors; ++v) {
      const float64x2_t b_vec = vld1q_f64(b_row + 2 * v);
      for (int q = 0; q < Rows; ++q) {
        acc[q][v] = vaddq_f64(acc[q][v], vmulq_f64(a_splat[q], b_vec));
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
    for (int v = 0; v < kVectors; ++v) vst1q_f64(c_row + 2 * v, acc[q][v]);
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
