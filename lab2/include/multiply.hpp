#pragma once

#include <vector>

#include "cache_info.hpp"
#include "matrix.hpp"

namespace lab2 {

void multiply_naive(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_vec(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_vec_ijk(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_reg(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_block_plain(const Matrix& a, const Matrix& b, Matrix& c, int tile);
void multiply_block(const Matrix& a, const Matrix& b, Matrix& c, int tile);
void multiply_block_multi(const Matrix& a, const Matrix& b, Matrix& c,
                          const std::vector<int>& tiles);

}
