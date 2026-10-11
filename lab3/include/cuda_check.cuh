#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace lab3 {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "CUDA: %s\n  %s\n  %s:%d\n", cudaGetErrorString(status), expr, file,
               line);
  std::exit(EXIT_FAILURE);
}

}

#define CUDA_CHECK(expr) ::lab3::cuda_check((expr), #expr, __FILE__, __LINE__)
