#include <cstring>

#include "block_matrix.hpp"
#include "multiply.hpp"

#ifndef MUL_FN
#define MUL_FN multiply_auto
#endif

namespace lab1 {

void MUL_FN(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c) {
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

        for (int r = 0; r < kBlk; ++r) {
          const Elem* const a_row = a_block + r * kBlk;
          Elem* const c_row = c_block + r * kBlk;

          Elem acc[kBlk];
          for (int col = 0; col < kBlk; ++col) acc[col] = c_row[col];

          for (int k = 0; k < kBlk; ++k) {
            const Elem a_value = a_row[k];
            const Elem* const b_row = b_block + k * kBlk;
            for (int col = 0; col < kBlk; ++col) acc[col] += a_value * b_row[col];
          }

          for (int col = 0; col < kBlk; ++col) c_row[col] = acc[col];
        }
      }
    }
  }
}

}
