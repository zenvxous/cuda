#pragma once

#include <cuda_runtime.h>

namespace lab3 {

struct JuliaParams {
  float cx = 0.0f;
  float cy = 0.0f;
  float x_min = 0.0f;
  float y_max = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  int width = 0;
  int height = 0;
  int max_iter = 0;
};

constexpr float kEscapeRadiusSq = 4.0f;

__host__ __device__ inline float2 grid_point(const JuliaParams& p, int col, int row) {
  return make_float2(p.x_min + static_cast<float>(col) * p.dx,
                     p.y_max - static_cast<float>(row) * p.dy);
}

__host__ __device__ inline int escape_time(float2 z, float cx, float cy, int max_iter) {
  for (int n = 0; n < max_iter; ++n) {
    const float xx = z.x * z.x;
    const float yy = z.y * z.y;
    if (xx + yy > kEscapeRadiusSq) return n;
    const float xy = z.x * z.y;
    z.y = xy + xy + cy;
    z.x = xx - yy + cx;
  }
  return max_iter;
}

}
