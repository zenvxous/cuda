#include "cache_info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace lab2 {
namespace {

constexpr std::size_t kElemSize = sizeof(double);
constexpr std::size_t kRequiredWidthBytes = 64;

#ifdef __APPLE__
std::size_t sysctl_size(const char* name) {
  std::uint64_t value = 0;
  std::size_t len = sizeof(value);
  if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) return 0;
  return static_cast<std::size_t>(value);
}
#else
std::size_t sysctl_size(const char*) { return 0; }
#endif

}

CacheInfo detect_caches() {
  CacheInfo info;

  const std::size_t line = sysctl_size("hw.cachelinesize");
  if (line != 0) info.line_size = line;

  info.l1d = sysctl_size("hw.perflevel0.l1dcachesize");
  if (info.l1d == 0) info.l1d = sysctl_size("hw.l1dcachesize");

  info.l2 = sysctl_size("hw.perflevel0.l2cachesize");
  if (info.l2 == 0) info.l2 = sysctl_size("hw.l2cachesize");

  info.l3 = sysctl_size("hw.perflevel0.l3cachesize");
  if (info.l3 == 0) info.l3 = sysctl_size("hw.l3cachesize");

  if (info.l1d == 0) info.l1d = 32u * 1024;
  if (info.l2 == 0) info.l2 = 1024u * 1024;

  if (info.l3 != 0) {
    info.llc = info.l3;
    info.llc_origin = "L3";
  } else {
    info.llc = info.l2;
    info.llc_origin = "L2 (L3 отсутствует)";
  }

  info.levels.clear();
  info.levels.push_back({1, info.l1d, "L1d"});
  info.levels.push_back({2, info.l2, "L2"});
  if (info.l3 != 0) info.levels.push_back({3, info.l3, "L3"});
  return info;
}

void apply_overrides(CacheInfo& info, std::size_t l1d, std::size_t l2, std::size_t l3,
                     std::size_t line) {
  if (line != 0) info.line_size = line;
  if (l1d != 0) info.l1d = l1d;
  if (l2 != 0) info.l2 = l2;
  if (l3 != 0) info.l3 = l3;

  if (info.l3 != 0) {
    info.llc = info.l3;
    info.llc_origin = (l3 != 0) ? "L3 (задан флагом)" : "L3";
  } else {
    info.llc = info.l2;
    info.llc_origin = (l2 != 0) ? "L2 (задан флагом)" : "L2 (L3 отсутствует)";
  }

  info.levels.clear();
  info.levels.push_back({1, info.l1d, "L1d"});
  info.levels.push_back({2, info.l2, "L2"});
  if (info.l3 != 0) info.levels.push_back({3, info.l3, "L3"});
}

std::size_t block_budget_bytes(std::size_t cache_size) {
  return static_cast<std::size_t>(static_cast<double>(cache_size) / 3.0 * 0.9);
}

int tile_side(std::size_t cache_size, std::size_t line_size) {
  const std::size_t budget = block_budget_bytes(cache_size);
  const double side = std::sqrt(static_cast<double>(budget) / static_cast<double>(kElemSize));

  std::size_t align_bytes = std::max<std::size_t>(line_size, kRequiredWidthBytes);
  if (align_bytes % kRequiredWidthBytes != 0) align_bytes = kRequiredWidthBytes;
  const std::size_t align_elems = align_bytes / kElemSize;

  std::size_t t = static_cast<std::size_t>(side);
  t = (t / align_elems) * align_elems;
  if (t == 0) t = align_elems;
  return static_cast<int>(t);
}

std::vector<CacheLevel> blocking_levels(const CacheInfo& info) {
  std::vector<CacheLevel> ordered;
  for (const CacheLevel& level : info.levels) {
    if (level.size != 0) ordered.push_back(level);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const CacheLevel& a, const CacheLevel& b) { return a.size > b.size; });

  std::vector<CacheLevel> result;
  for (const CacheLevel& level : ordered) {
    if (!result.empty() && tile_side(level.size, info.line_size) ==
                               tile_side(result.back().size, info.line_size)) {
      continue;
    }
    result.push_back(level);
  }
  return result;
}

}
