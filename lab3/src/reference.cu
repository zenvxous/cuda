#include <cstddef>

#include "kernels.cuh"

namespace lab3 {

void julia_reference(const JuliaParams& p, std::uint16_t* out) {
  for (int row = 0; row < p.height; ++row) {
    std::uint16_t* out_row = out + static_cast<std::size_t>(row) * p.width;
    for (int col = 0; col < p.width; ++col) {
      out_row[col] = static_cast<std::uint16_t>(
          escape_time(grid_point(p, col, row), p.cx, p.cy, p.max_iter));
    }
  }
}

}
