#pragma once

#include <arm_neon.h>
#include <cstddef>

#include "matrix.hpp"

namespace lab2 {

inline void kernel_row(const Elem* __restrict a_row, const Elem* __restrict b, int b_stride,
                       Elem* __restrict c_row, int k0, int k1, int j0, int j1) {
  for (int k = k0; k < k1; ++k) {
    const Elem a_value = a_row[k];
    const float64x2_t a = vdupq_n_f64(a_value);
    const Elem* const b_row = b + static_cast<std::size_t>(k) * b_stride;
    int j = j0;
    for (; j + 2 <= j1; j += 2) {
      const float64x2_t acc = vld1q_f64(c_row + j);
      vst1q_f64(c_row + j, vaddq_f64(acc, vmulq_f64(a, vld1q_f64(b_row + j))));
    }
    for (; j < j1; ++j) c_row[j] += a_value * b_row[j];
  }
}

inline void kernel_row_pair(const Elem* __restrict a_row0, const Elem* __restrict a_row1,
                            const Elem* __restrict b, int b_stride, Elem* __restrict c_row0,
                            Elem* __restrict c_row1, int k0, int k1, int j0, int j1) {
  for (int k = k0; k < k1; ++k) {
    const Elem a_value0 = a_row0[k];
    const Elem a_value1 = a_row1[k];
    const float64x2_t a0 = vdupq_n_f64(a_value0);
    const float64x2_t a1 = vdupq_n_f64(a_value1);
    const Elem* const b_row = b + static_cast<std::size_t>(k) * b_stride;
    int j = j0;
    for (; j + 2 <= j1; j += 2) {
      const float64x2_t b_vec = vld1q_f64(b_row + j);
      vst1q_f64(c_row0 + j, vaddq_f64(vld1q_f64(c_row0 + j), vmulq_f64(a0, b_vec)));
      vst1q_f64(c_row1 + j, vaddq_f64(vld1q_f64(c_row1 + j), vmulq_f64(a1, b_vec)));
    }
    for (; j < j1; ++j) {
      const Elem b_value = b_row[j];
      c_row0[j] += a_value0 * b_value;
      c_row1[j] += a_value1 * b_value;
    }
  }
}

inline void kernel_ikj(const Matrix& a, const Matrix& b, Matrix& c, int i0, int i1, int k0,
                       int k1, int j0, int j1) {
  const Elem* const b_data = b.data();
  const int b_stride = b.stride();

  int i = i0;
  for (; i + 2 <= i1; i += 2) {
    kernel_row_pair(a.row(i), a.row(i + 1), b_data, b_stride, c.row(i), c.row(i + 1), k0, k1,
                    j0, j1);
  }
  for (; i < i1; ++i) kernel_row(a.row(i), b_data, b_stride, c.row(i), k0, k1, j0, j1);
}


inline void kernel_reg(const Matrix& a, const Matrix& b, Matrix& c, int i0, int i1, int k0,
                       int k1, int j0, int j1) {
  constexpr int kRows = 4;
  constexpr int kVecs = 4;
  constexpr int kCols = kVecs * 2;

  const Elem* const b_data = b.data();
  const int b_stride = b.stride();

  int i = i0;
  for (; i + kRows <= i1; i += kRows) {
    const Elem* a_rows[kRows];
    Elem* c_rows[kRows];
    for (int q = 0; q < kRows; ++q) {
      a_rows[q] = a.row(i + q);
      c_rows[q] = c.row(i + q);
    }

    int j = j0;
    for (; j + kCols <= j1; j += kCols) {
      float64x2_t acc[kRows][kVecs];
      for (int q = 0; q < kRows; ++q) {
        for (int v = 0; v < kVecs; ++v) acc[q][v] = vld1q_f64(c_rows[q] + j + 2 * v);
      }

      for (int k = k0; k < k1; ++k) {
        const Elem* const b_row = b_data + static_cast<std::size_t>(k) * b_stride + j;
        float64x2_t b_vec[kVecs];
        for (int v = 0; v < kVecs; ++v) b_vec[v] = vld1q_f64(b_row + 2 * v);
        for (int q = 0; q < kRows; ++q) {
          const float64x2_t a_vec = vdupq_n_f64(a_rows[q][k]);
          for (int v = 0; v < kVecs; ++v) {
            acc[q][v] = vaddq_f64(acc[q][v], vmulq_f64(a_vec, b_vec[v]));
          }
        }
      }

      for (int q = 0; q < kRows; ++q) {
        for (int v = 0; v < kVecs; ++v) vst1q_f64(c_rows[q] + j + 2 * v, acc[q][v]);
      }
    }
    if (j < j1) kernel_ikj(a, b, c, i, i + kRows, k0, k1, j, j1);
  }
  if (i < i1) kernel_ikj(a, b, c, i, i1, k0, k1, j0, j1);
}

}
