#include <algorithm>
#include <cstring>

#include "kernel.hpp"
#include "multiply.hpp"

namespace lab2 {

namespace {

template <typename Kernel>
void blocked(const Matrix& a, const Matrix& b, Matrix& c, int tile, Kernel kernel) {
  const int m = a.rows();
  const int k_size = a.cols();
  const int n = b.cols();

  std::memset(c.data(), 0, c.bytes());

  for (int ii = 0; ii < m; ii += tile) {
    const int i_end = std::min(ii + tile, m);
    for (int kk = 0; kk < k_size; kk += tile) {
      const int k_end = std::min(kk + tile, k_size);
      for (int jj = 0; jj < n; jj += tile) {
        const int j_end = std::min(jj + tile, n);
        kernel(a, b, c, ii, i_end, kk, k_end, jj, j_end);
      }
    }
  }
}

}

void multiply_block(const Matrix& a, const Matrix& b, Matrix& c, int tile) {
  blocked(a, b, c, tile,
          [](const Matrix& x, const Matrix& y, Matrix& z, int i0, int i1, int k0, int k1, int j0,
             int j1) { kernel_reg(x, y, z, i0, i1, k0, k1, j0, j1); });
}

void multiply_block_plain(const Matrix& a, const Matrix& b, Matrix& c, int tile) {
  blocked(a, b, c, tile,
          [](const Matrix& x, const Matrix& y, Matrix& z, int i0, int i1, int k0, int k1, int j0,
             int j1) { kernel_ikj(x, y, z, i0, i1, k0, k1, j0, j1); });
}

}
