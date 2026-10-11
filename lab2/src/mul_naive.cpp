#include <cstring>

#include "multiply.hpp"

namespace lab2 {

void multiply_naive(const Matrix& a, const Matrix& b, Matrix& c) {
  const int m = a.rows();
  const int k_size = a.cols();
  const int n = b.cols();

  for (int i = 0; i < m; ++i) {
    const Elem* const a_row = a.row(i);
    Elem* const c_row = c.row(i);
    for (int j = 0; j < n; ++j) {
      Elem sum = Elem{0};
      for (int k = 0; k < k_size; ++k) sum += a_row[k] * b.row(k)[j];
      c_row[j] = sum;
    }
  }
}

}
