#pragma once

#include <cstddef>
#include <cstdint>

namespace lab2 {

using Elem = double;

constexpr std::size_t kAlignment = 128;
constexpr int kStrideAlignElems = 16;

class Matrix {
 public:
  Matrix() = default;
  Matrix(int rows, int cols);
  ~Matrix();

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int stride() const { return stride_; }

  std::size_t elem_count() const { return static_cast<std::size_t>(rows_) * stride_; }
  std::size_t bytes() const { return elem_count() * sizeof(Elem); }
  std::size_t payload_bytes() const {
    return static_cast<std::size_t>(rows_) * cols_ * sizeof(Elem);
  }

  Elem* data() { return data_; }
  const Elem* data() const { return data_; }

  Elem* row(int i) { return data_ + static_cast<std::size_t>(i) * stride_; }
  const Elem* row(int i) const { return data_ + static_cast<std::size_t>(i) * stride_; }

 private:
  int rows_ = 0;
  int cols_ = 0;
  int stride_ = 0;
  Elem* data_ = nullptr;
};

struct CompareResult {
  bool bit_exact = false;
  double max_abs_diff = 0.0;
  std::size_t diff_count = 0;
};

void fill_random(Matrix& m, std::uint64_t seed);
void fill_zero(Matrix& m);
CompareResult compare(const Matrix& a, const Matrix& b);
double checksum(const Matrix& m);

}
