#pragma once

#include <chrono>
#include <cstdint>
#include <utility>

namespace lab1 {

using Clock = std::chrono::high_resolution_clock;

struct Measurement {
  double seconds = 0.0;
  std::uint64_t ticks = 0;
};

inline std::uint64_t read_tick_counter() {
  std::uint64_t value;
  asm volatile("mrs %0, cntvct_el0" : "=r"(value));
  return value;
}

inline double tick_counter_hz() {
  std::uint64_t freq;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
  return static_cast<double>(freq);
}

template <typename Fn>
Measurement measure(Fn&& fn) {
  const auto wall_begin = Clock::now();
  const std::uint64_t ticks_begin = read_tick_counter();

  std::forward<Fn>(fn)();

  const std::uint64_t ticks_end = read_tick_counter();
  const auto wall_end = Clock::now();

  return {std::chrono::duration<double>(wall_end - wall_begin).count(),
          ticks_end - ticks_begin};
}

}
