#pragma once

#include <chrono>
#include <cstdint>
#include <utility>
#include <x86intrin.h>

namespace lab1 {

using Clock = std::chrono::high_resolution_clock;

struct Measurement {
  double seconds = 0.0;
  std::uint64_t ticks = 0;
};

template <typename Fn>
Measurement measure(Fn&& fn) {
  const auto wall_begin = Clock::now();
  const std::uint64_t ticks_begin = __rdtsc();

  std::forward<Fn>(fn)();

  const std::uint64_t ticks_end = __rdtsc();
  const auto wall_end = Clock::now();

  return {std::chrono::duration<double>(wall_end - wall_begin).count(),
          ticks_end - ticks_begin};
}

}
