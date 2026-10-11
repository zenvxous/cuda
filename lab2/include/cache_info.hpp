#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace lab2 {

struct CacheLevel {
  int level = 0;
  std::size_t size = 0;
  std::string origin;
};

struct CacheInfo {
  std::size_t line_size = 64;
  std::size_t l1d = 0;
  std::size_t l2 = 0;
  std::size_t l3 = 0;
  std::size_t llc = 0;
  std::string llc_origin;
  std::vector<CacheLevel> levels;
};

CacheInfo detect_caches();

void apply_overrides(CacheInfo& info, std::size_t l1d, std::size_t l2, std::size_t l3,
                     std::size_t line);

std::size_t block_budget_bytes(std::size_t cache_size);

int tile_side(std::size_t cache_size, std::size_t line_size);

std::vector<CacheLevel> blocking_levels(const CacheInfo& info);

}
