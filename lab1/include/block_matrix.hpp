#pragma once

#include <cstddef>
#include <cstdint>

#ifndef BLK
#define BLK 8
#endif

namespace lab1 {

using Elem = double;

constexpr int kBlk = BLK;
constexpr int kBlkArea = kBlk * kBlk;

class BlockMatrix {
 public:
  BlockMatrix() = default;
  BlockMatrix(int rows, int cols);
  ~BlockMatrix();

  BlockMatrix(const BlockMatrix&) = delete;
  BlockMatrix& operator=(const BlockMatrix&) = delete;

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  std::size_t block_count() const { return static_cast<std::size_t>(rows_) * cols_; }
  std::size_t elem_count() const { return block_count() * kBlkArea; }
  std::size_t bytes() const { return elem_count() * sizeof(Elem); }

  Elem* data() { return data_; }
  const Elem* data() const { return data_; }

  Elem* block(int i, int j) {
    return data_ + (static_cast<std::size_t>(i) * cols_ + j) * kBlkArea;
  }
  const Elem* block(int i, int j) const {
    return data_ + (static_cast<std::size_t>(i) * cols_ + j) * kBlkArea;
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  Elem* data_ = nullptr;
};

struct CompareResult {
  bool bit_exact = false;
  double max_abs_diff = 0.0;
  std::size_t diff_count = 0;
};

void fill_random(BlockMatrix& m, std::uint64_t seed);
void fill_zero(BlockMatrix& m);
CompareResult compare(const BlockMatrix& a, const BlockMatrix& b);
double checksum(const BlockMatrix& m);

}
