#include <cstring>

#include "kernel.hpp"
#include "multiply.hpp"

namespace lab2 {

void multiply_vec(const Matrix& a, const Matrix& b, Matrix& c) {
  std::memset(c.data(), 0, c.bytes());
  kernel_ikj(a, b, c, 0, a.rows(), 0, a.cols(), 0, b.cols());
}

void multiply_reg(const Matrix& a, const Matrix& b, Matrix& c) {
  std::memset(c.data(), 0, c.bytes());
  kernel_reg(a, b, c, 0, a.rows(), 0, a.cols(), 0, b.cols());
}

}
