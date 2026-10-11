#include <cstddef>

#include "cuda_check.cuh"
#include "kernels.cuh"

namespace lab3 {

namespace {

__global__ void make_grid_kernel(float2* __restrict__ grid, JuliaParams p) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= p.width || row >= p.height) return;
  grid[static_cast<std::size_t>(row) * p.width + col] = grid_point(p, col, row);
}

__global__ void julia_grid_kernel(const float2* __restrict__ grid,
                                  std::uint16_t* __restrict__ out, JuliaParams p) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= p.width || row >= p.height) return;
  const std::size_t idx = static_cast<std::size_t>(row) * p.width + col;
  out[idx] = static_cast<std::uint16_t>(escape_time(grid[idx], p.cx, p.cy, p.max_iter));
}

__global__ void julia_fused_kernel(std::uint16_t* __restrict__ out, JuliaParams p) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= p.width || row >= p.height) return;
  const std::size_t idx = static_cast<std::size_t>(row) * p.width + col;
  out[idx] = static_cast<std::uint16_t>(
      escape_time(grid_point(p, col, row), p.cx, p.cy, p.max_iter));
}

dim3 grid_for(const JuliaParams& p, dim3 block) {
  return dim3((p.width + block.x - 1) / block.x, (p.height + block.y - 1) / block.y);
}

}

void launch_make_grid(float2* grid, const JuliaParams& p, dim3 block) {
  make_grid_kernel<<<grid_for(p, block), block>>>(grid, p);
  CUDA_CHECK(cudaGetLastError());
}

void launch_julia_from_grid(const float2* grid, std::uint16_t* out, const JuliaParams& p,
                            dim3 block) {
  julia_grid_kernel<<<grid_for(p, block), block>>>(grid, out, p);
  CUDA_CHECK(cudaGetLastError());
}

void launch_julia_fused(std::uint16_t* out, const JuliaParams& p, dim3 block) {
  julia_fused_kernel<<<grid_for(p, block), block>>>(out, p);
  CUDA_CHECK(cudaGetLastError());
}

}
