#include "cache_probe.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>
#include <numeric>
#include <random>

#include "timer.hpp"

namespace lab2 {

std::vector<ProbePoint> probe_cache(std::size_t line_size, std::size_t max_bytes) {
  std::vector<ProbePoint> points;
  const std::size_t stride = line_size;
  std::mt19937_64 rng(12345);

  for (std::size_t bytes = 4096; bytes <= max_bytes; bytes *= 2) {
    const std::size_t slots = bytes / stride;
    if (slots < 4) continue;

    void* raw = nullptr;
    if (posix_memalign(&raw, stride, slots * stride) != 0) throw std::bad_alloc();
    char* const buffer = static_cast<char*>(raw);

    std::vector<std::size_t> order(slots);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = slots - 1; i > 0; --i) {
      std::uniform_int_distribution<std::size_t> dist(0, i);
      std::swap(order[i], order[dist(rng)]);
    }
    for (std::size_t i = 0; i < slots; ++i) {
      void** const slot = reinterpret_cast<void**>(buffer + order[i] * stride);
      *slot = buffer + order[(i + 1) % slots] * stride;
    }

    std::size_t accesses = slots * 16;
    if (accesses < 2000000) accesses = 2000000;

    void* pointer = buffer;
    const auto begin = Clock::now();
    for (std::size_t i = 0; i < accesses; ++i) pointer = *reinterpret_cast<void**>(pointer);
    const auto end = Clock::now();

    static void* sink;
    sink = pointer;
    (void)sink;

    const double seconds = std::chrono::duration<double>(end - begin).count();
    points.push_back({bytes, seconds * 1e9 / static_cast<double>(accesses)});
    std::free(raw);
  }
  return points;
}

}
