#include <algorithm>
#include <cstring>

#include "kernel.hpp"
#include "multiply.hpp"

namespace lab2 {
namespace {

struct Range {
  int i0, i1, k0, k1, j0, j1;
};

void recurse(const Matrix& a, const Matrix& b, Matrix& c, const std::vector<int>& tiles,
             std::size_t level, const Range& r) {
  if (level == tiles.size()) {
    kernel_reg(a, b, c, r.i0, r.i1, r.k0, r.k1, r.j0, r.j1);
    return;
  }

  const int tile = tiles[level];
  for (int ii = r.i0; ii < r.i1; ii += tile) {
    const int i_end = std::min(ii + tile, r.i1);
    for (int kk = r.k0; kk < r.k1; kk += tile) {
      const int k_end = std::min(kk + tile, r.k1);
      for (int jj = r.j0; jj < r.j1; jj += tile) {
        const int j_end = std::min(jj + tile, r.j1);
        recurse(a, b, c, tiles, level + 1, Range{ii, i_end, kk, k_end, jj, j_end});
      }
    }
  }
}

}

void multiply_block_multi(const Matrix& a, const Matrix& b, Matrix& c,
                          const std::vector<int>& tiles) {
  std::memset(c.data(), 0, c.bytes());
  recurse(a, b, c, tiles, 0, Range{0, a.rows(), 0, a.cols(), 0, b.cols()});
}

}
