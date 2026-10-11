#include <arm_neon.h>
#include <cstring>

#include "multiply.hpp"

namespace lab2 {

void multiply_vec_ijk(const Matrix& a, const Matrix& b, Matrix& c) {
  const int m = a.rows();
  const int k_size = a.cols();
  const int n = b.cols();
  const Elem* const b_data = b.data();
  const int b_stride = b.stride();

  std::memset(c.data(), 0, c.bytes());

  for (int i = 0; i < m; ++i) {
    const Elem* const __restrict a_row = a.row(i);
    Elem* const __restrict c_row = c.row(i);

    int j = 0;
    for (; j + 2 <= n; j += 2) {
      float64x2_t acc = vdupq_n_f64(0.0);
      for (int k = 0; k < k_size; ++k) {
        const float64x2_t a_vec = vdupq_n_f64(a_row[k]);
        const Elem* const b_row = b_data + static_cast<std::size_t>(k) * b_stride + j;
        acc = vaddq_f64(acc, vmulq_f64(a_vec, vld1q_f64(b_row)));
      }
      vst1q_f64(c_row + j, acc);
    }
    for (; j < n; ++j) {
      Elem sum = Elem{0};
      for (int k = 0; k < k_size; ++k) {
        sum += a_row[k] * b_data[static_cast<std::size_t>(k) * b_stride + j];
      }
      c_row[j] = sum;
    }
  }
}

}
