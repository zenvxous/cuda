#pragma once

#include <cstddef>
#include <vector>

namespace lab2 {

struct ProbePoint {
  std::size_t bytes = 0;
  double ns_per_access = 0.0;
};

std::vector<ProbePoint> probe_cache(std::size_t line_size, std::size_t max_bytes);

}
