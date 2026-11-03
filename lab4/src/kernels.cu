#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda_check.cuh"
#include "kernels.cuh"

namespace lab4 {

namespace {

struct TermAt {
  const float* x;
  const float* w;
  __host__ __device__ Pair operator()(int i) const { return make_term(x[i], w[i]); }
};

struct PairAt {
  const Pair* p;
  __device__ Pair operator()(int i) const { return p[i]; }
};

__device__ inline Pair shfl_down(Pair v, unsigned offset) {
  return Pair{__shfl_down_sync(0xFFFFFFFFu, v.wx, offset),
              __shfl_down_sync(0xFFFFFFFFu, v.w, offset)};
}

template <typename Src>
__global__ void interleaved_kernel(Src src, int n, Pair* __restrict__ out) {
  extern __shared__ Pair sh[];
  const unsigned tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  sh[tid] = i < n ? src(i) : zero_pair();
  __syncthreads();
  for (unsigned s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) sh[tid] = combine(sh[tid], sh[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sh[0];
}

template <typename Src>
__global__ void sequential_kernel(Src src, int n, Pair* __restrict__ out) {
  extern __shared__ Pair sh[];
  const unsigned tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  sh[tid] = i < n ? src(i) : zero_pair();
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sh[tid] = combine(sh[tid], sh[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sh[0];
}

template <typename Src>
__global__ void grid_stride_kernel(Src src, int n, Pair* __restrict__ out) {
  extern __shared__ Pair sh[];
  const unsigned tid = threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  Pair acc = zero_pair();
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) acc = combine(acc, src(i));
  sh[tid] = acc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s) sh[tid] = combine(sh[tid], sh[tid + s]);
    __syncthreads();
  }
  if (tid >= 32) return;
  Pair v = sh[tid];
  for (unsigned offset = 16; offset > 0; offset >>= 1) v = combine(v, shfl_down(v, offset));
  if (tid == 0) out[blockIdx.x] = v;
}

template <typename Src>
void launch_level(Tree tree, Src src, int n, int block, Pair* out) {
  const int blocks = div_up(n, block);
  const std::size_t shared = block * sizeof(Pair);
  if (tree == Tree::interleaved) {
    interleaved_kernel<<<blocks, block, shared>>>(src, n, out);
  } else {
    sequential_kernel<<<blocks, block, shared>>>(src, n, out);
  }
  CUDA_CHECK(cudaGetLastError());
}

auto term_iterator(const float* x, const float* w) {
  return thrust::make_transform_iterator(thrust::counting_iterator<int>(0), TermAt{x, w});
}

}

std::size_t cub_temp_bytes(int n) {
  std::size_t bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, bytes, term_iterator(nullptr, nullptr),
                                       static_cast<Pair*>(nullptr), n, Combine{}, zero_pair()));
  return bytes;
}

void launch_tree(Tree tree, const float* x, const float* w, int n, int block, const Workspace& ws) {
  int count = div_up(n, block);
  Pair* dst = count == 1 ? ws.result : ws.a;
  launch_level(tree, TermAt{x, w}, n, block, dst);
  while (count > 1) {
    const Pair* src = dst;
    const int next = div_up(count, block);
    dst = next == 1 ? ws.result : (src == ws.a ? ws.b : ws.a);
    launch_level(tree, PairAt{src}, count, block, dst);
    count = next;
  }
}

void launch_grid_stride(const float* x, const float* w, int n, int block, int grid,
                        const Workspace& ws) {
  const std::size_t shared = block * sizeof(Pair);
  grid_stride_kernel<<<grid, block, shared>>>(TermAt{x, w}, n, grid == 1 ? ws.result : ws.a);
  CUDA_CHECK(cudaGetLastError());
  if (grid == 1) return;
  grid_stride_kernel<<<1, block, shared>>>(PairAt{ws.a}, grid, ws.result);
  CUDA_CHECK(cudaGetLastError());
}

void launch_cub(const float* x, const float* w, int n, const Workspace& ws) {
  std::size_t bytes = ws.cub_bytes;
  CUDA_CHECK(cub::DeviceReduce::Reduce(ws.cub_temp, bytes, term_iterator(x, w), ws.result, n,
                                       Combine{}, zero_pair()));
}

}
