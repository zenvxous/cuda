#include "image.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <vector>

namespace lab3 {

namespace {

unsigned char channel(double value) {
  return static_cast<unsigned char>(std::clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

}

bool write_ppm(const std::string& path, const std::uint16_t* counts, int width, int height,
               int max_iter) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file << "P6\n" << width << ' ' << height << "\n255\n";

  std::vector<unsigned char> line(static_cast<std::size_t>(width) * 3);
  for (int row = 0; row < height; ++row) {
    const std::uint16_t* src = counts + static_cast<std::size_t>(row) * width;
    for (int col = 0; col < width; ++col) {
      unsigned char* px = line.data() + static_cast<std::size_t>(col) * 3;
      if (src[col] >= max_iter) {
        px[0] = px[1] = px[2] = 0;
        continue;
      }
      const double t = static_cast<double>(src[col]) / max_iter;
      const double s = 1.0 - t;
      px[0] = channel(9.0 * s * t * t * t);
      px[1] = channel(15.0 * s * s * t * t);
      px[2] = channel(8.5 * s * s * s * t);
    }
    file.write(reinterpret_cast<const char*>(line.data()),
               static_cast<std::streamsize>(line.size()));
  }
  return static_cast<bool>(file);
}

}
