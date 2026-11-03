#pragma once

#include <algorithm>
#include <cstddef>

#include "wmean.cuh"

namespace lab4 {

enum class Tree { interleaved, sequential };

struct Workspace {
  Pair* a = nullptr;
  Pair* b = nullptr;
  Pair* result = nullptr;
  void* cub_temp = nullptr;
  std::size_t cub_bytes = 0;
};

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline int grid_stride_blocks(int n, int block, int cap) { return std::min(div_up(n, block), cap); }

inline std::size_t workspace_pairs(int n) { return static_cast<std::size_t>(div_up(n, 32)); }

std::size_t cub_temp_bytes(int n);

void launch_tree(Tree tree, const float* x, const float* w, int n, int block, const Workspace& ws);
void launch_grid_stride(const float* x, const float* w, int n, int block, int grid,
                        const Workspace& ws);
void launch_cub(const float* x, const float* w, int n, const Workspace& ws);

Pair reduce_sequential(const float* x, const float* w, int n);
double weighted_mean_exact(const float* x, const float* w, int n);
Pair emulate_tree(Tree tree, const float* x, const float* w, int n, int block);
Pair emulate_grid_stride(const float* x, const float* w, int n, int block, int grid);

}
