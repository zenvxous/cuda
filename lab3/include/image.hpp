#pragma once

#include <cstdint>
#include <string>

namespace lab3 {

bool write_ppm(const std::string& path, const std::uint16_t* counts, int width, int height,
               int max_iter);

}
