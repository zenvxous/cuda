#include <vector>

#include "kernels.cuh"

namespace lab4 {

namespace {

std::vector<Pair> make_terms(const float* x, const float* w, int n) {
  std::vector<Pair> terms(n);
  for (int i = 0; i < n; ++i) terms[i] = make_term(x[i], w[i]);
  return terms;
}

void tree_interleaved(Pair* sh, int block) {
  for (int s = 1; s < block; s *= 2) {
    for (int tid = 0; tid < block; tid += 2 * s) sh[tid] = combine(sh[tid], sh[tid + s]);
  }
}

void tree_sequential(Pair* sh, int block) {
  for (int s = block / 2; s > 0; s >>= 1) {
    for (int tid = 0; tid < s; ++tid) sh[tid] = combine(sh[tid], sh[tid + s]);
  }
}

std::vector<Pair> tree_level(Tree tree, const std::vector<Pair>& in, int block) {
  const int n = static_cast<int>(in.size());
  const int blocks = div_up(n, block);
  std::vector<Pair> out(blocks);
  std::vector<Pair> sh(block);
  for (int b = 0; b < blocks; ++b) {
    for (int t = 0; t < block; ++t) {
      const int i = b * block + t;
      sh[t] = i < n ? in[i] : zero_pair();
    }
    if (tree == Tree::interleaved) {
      tree_interleaved(sh.data(), block);
    } else {
      tree_sequential(sh.data(), block);
    }
    out[b] = sh[0];
  }
  return out;
}

std::vector<Pair> grid_stride_level(const std::vector<Pair>& in, int block, int grid) {
  const int n = static_cast<int>(in.size());
  const int stride = grid * block;
  std::vector<Pair> out(grid);
  std::vector<Pair> sh(block);
  for (int b = 0; b < grid; ++b) {
    for (int t = 0; t < block; ++t) {
      Pair acc = zero_pair();
      for (int i = b * block + t; i < n; i += stride) acc = combine(acc, in[i]);
      sh[t] = acc;
    }
    tree_sequential(sh.data(), block);
    out[b] = sh[0];
  }
  return out;
}

}

Pair reduce_sequential(const float* x, const float* w, int n) {
  Pair acc = zero_pair();
  for (int i = 0; i < n; ++i) acc = combine(acc, make_term(x[i], w[i]));
  return acc;
}

double weighted_mean_exact(const float* x, const float* w, int n) {
  double sum_wx = 0.0;
  double sum_w = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_wx += static_cast<double>(w[i]) * static_cast<double>(x[i]);
    sum_w += static_cast<double>(w[i]);
  }
  return sum_wx / sum_w;
}

Pair emulate_tree(Tree tree, const float* x, const float* w, int n, int block) {
  std::vector<Pair> v = make_terms(x, w, n);
  do {
    v = tree_level(tree, v, block);
  } while (v.size() > 1);
  return v[0];
}

Pair emulate_grid_stride(const float* x, const float* w, int n, int block, int grid) {
  std::vector<Pair> v = grid_stride_level(make_terms(x, w, n), block, grid);
  if (grid > 1) v = grid_stride_level(v, block, 1);
  return v[0];
}

}
