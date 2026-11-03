#pragma once

#include <cuda_runtime.h>

namespace lab4 {

struct alignas(8) Pair {
  float wx;
  float w;
};

__host__ __device__ inline Pair zero_pair() { return Pair{0.0f, 0.0f}; }

__host__ __device__ inline Pair make_term(float x, float w) { return Pair{w * x, w}; }

__host__ __device__ inline Pair combine(Pair a, Pair b) { return Pair{a.wx + b.wx, a.w + b.w}; }

struct Combine {
  __host__ __device__ Pair operator()(const Pair& a, const Pair& b) const { return combine(a, b); }
};

}
