#pragma once

#include "block_matrix.hpp"

namespace lab1 {

void multiply_auto(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);
void multiply_novec(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);
void multiply_simd(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);

}
