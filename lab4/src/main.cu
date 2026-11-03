#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "kernels.cuh"

namespace {

using lab4::Pair;
using lab4::Tree;

struct Options {
  int n = 1000000;
  unsigned seed = 2026;
  int repeat = 20;
  int tree_block = 256;
  std::vector<int> blocks = {128, 256, 512, 1024};
};

struct Device {
  int sm_count = 0;
  int threads_per_sm = 0;
  int l2_bytes = 0;
};

enum class Check { none, bitwise, tolerance };

struct Row {
  std::string title;
  double best_ms = 0.0;
  double mean_ms = 0.0;
  Pair value{};
  Pair expected{};
  Check check = Check::none;
  bool is_cpu = false;
};

constexpr double kCubTolerance = 1e-5;

std::size_t visual_width(const std::string& text) {
  std::size_t width = 0;
  for (const unsigned char ch : text) {
    if ((ch & 0xC0) != 0x80) ++width;
  }
  return width;
}

std::string pad_right(const std::string& text, std::size_t width) {
  const std::size_t current = visual_width(text);
  return current >= width ? text : text + std::string(width - current, ' ');
}

std::string pad_left(const std::string& text, std::size_t width) {
  const std::size_t current = visual_width(text);
  return current >= width ? text : std::string(width - current, ' ') + text;
}

std::string rule(std::size_t width) {
  std::string line;
  for (std::size_t i = 0; i < width; ++i) line += "─";
  return line;
}

std::string fmt(const char* format, double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), format, value);
  return buf;
}

void print_usage(const char* program) {
  std::printf(
      "Лабораторная работа №4, задание 10: взвешенное среднее через reduce на CUDA.\n\n"
      "Использование: %s [ключи]\n\n"
      "  --n N         размер массивов x и w, N ∈ [10, 1000000] (по умолчанию 1000000)\n"
      "  --seed S      зерно генератора данных (по умолчанию 2026)\n"
      "  --repeat R    число замеров каждой версии (по умолчанию 20)\n"
      "  --block B     размер блока для всех ядер, степень двойки от 32 до 1024\n"
      "                (по умолчанию G1, G2 — 256, G3 — 128, 256, 512, 1024)\n"
      "  -h, --help    эта справка\n",
      program);
}

int parse_int(const std::string& text, const char* name) {
  std::size_t used = 0;
  int value = 0;
  try {
    value = std::stoi(text, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != text.size()) {
    throw std::invalid_argument(std::string("неверное целое для ") + name + ": " + text);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options opt;
  auto need = [&](int& i, int count, const std::string& key) {
    if (i + count >= argc) throw std::invalid_argument("не хватает значений для " + key);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--n") {
      need(i, 1, arg);
      opt.n = parse_int(argv[++i], "N");
    } else if (arg == "--seed") {
      need(i, 1, arg);
      const int seed = parse_int(argv[++i], "--seed");
      if (seed < 0) throw std::invalid_argument("--seed должно быть неотрицательным");
      opt.seed = static_cast<unsigned>(seed);
    } else if (arg == "--repeat") {
      need(i, 1, arg);
      opt.repeat = parse_int(argv[++i], "--repeat");
    } else if (arg == "--block") {
      need(i, 1, arg);
      const int block = parse_int(argv[++i], "--block");
      if (block < 32 || block > 1024 || (block & (block - 1)) != 0) {
        throw std::invalid_argument("--block должно быть степенью двойки от 32 до 1024");
      }
      opt.tree_block = block;
      opt.blocks = {block};
    } else {
      throw std::invalid_argument("неизвестный ключ: " + arg);
    }
  }

  if (opt.n < 10 || opt.n > 1000000) throw std::invalid_argument("N должно лежать в [10, 10^6]");
  if (opt.repeat < 1) throw std::invalid_argument("--repeat должно быть не меньше 1");
  return opt;
}

Device print_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime = 0;
  int driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  int clock_khz = 0;
  int bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));
  const double peak_gbs = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;

  std::printf("GPU:        %s, CC %d.%d, %d SM, %.1f ГБ, L2 %.0f МБ\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount,
              prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.l2CacheSize / 1048576.0);
  std::printf("Память:     шина %d бит, пиковая полоса %.0f ГБ/с\n", bus_bits, peak_gbs);
  std::printf("CUDA:       runtime %d.%d, driver %d.%d\n", runtime / 1000, runtime % 1000 / 10,
              driver / 1000, driver % 1000 / 10);
  return Device{prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.l2CacheSize};
}

template <typename Fn>
void time_gpu(Fn&& fn, int repeat, Row& row, void* flush, std::size_t flush_bytes) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  fn();
  CUDA_CHECK(cudaDeviceSynchronize());

  double best = std::numeric_limits<double>::infinity();
  double sum = 0.0;
  for (int r = 0; r < repeat; ++r) {
    if (flush_bytes > 0) CUDA_CHECK(cudaMemsetAsync(flush, r & 0xFF, flush_bytes));
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    best = std::min(best, static_cast<double>(ms));
    sum += ms;
  }
  row.best_ms = best;
  row.mean_ms = sum / repeat;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

double mean_of(const Pair& p) { return static_cast<double>(p.wx / p.w); }

double rel_error(const Pair& p, double exact) { return std::fabs(mean_of(p) - exact) / exact; }

bool same_bits(const Pair& a, const Pair& b) { return std::memcmp(&a, &b, sizeof(Pair)) == 0; }

}

int main(int argc, char** argv) {
  Options opt;
  try {
    opt = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Ошибка: %s\nПодсказка: %s --help\n", e.what(), argv[0]);
    return EXIT_FAILURE;
  }

  const int n = opt.n;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  const Device dev = print_device();
  std::printf("Задача:     x̄_w = Σ wᵢxᵢ / Σ wᵢ, N = %d, x ∈ [0, 100], w ∈ [0.01, 10], "
              "seed = %u\n",
              n, opt.seed);
  std::printf("Данные:     %.1f МБ (x и w float), замеров каждой версии: %d\n\n",
              2.0 * bytes / (1024.0 * 1024.0), opt.repeat);

  float* h_x = nullptr;
  float* h_w = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_x, bytes));
  CUDA_CHECK(cudaMallocHost(&h_w, bytes));
  std::mt19937 gen(opt.seed);
  std::uniform_real_distribution<float> grade(0.0f, 100.0f);
  std::uniform_real_distribution<float> weight(0.01f, 10.0f);
  for (int i = 0; i < n; ++i) {
    h_x[i] = grade(gen);
    h_w[i] = weight(gen);
  }
  if (!std::all_of(h_w, h_w + n, [](float w) { return w > 0.0f; })) {
    std::fprintf(stderr, "Ошибка: веса должны быть положительными\n");
    return EXIT_FAILURE;
  }

  const double exact = lab4::weighted_mean_exact(h_x, h_w, n);
  std::vector<Row> rows;

  {
    Row row;
    row.title = "CPU, 1 поток, float подряд";
    row.is_cpu = true;
    double best = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (int r = 0; r < opt.repeat; ++r) {
      const auto begin = std::chrono::high_resolution_clock::now();
      row.value = lab4::reduce_sequential(h_x, h_w, n);
      const auto end = std::chrono::high_resolution_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
      best = std::min(best, ms);
      sum += ms;
    }
    row.best_ms = best;
    row.mean_ms = sum / opt.repeat;
    rows.push_back(row);
  }

  float* d_x = nullptr;
  float* d_w = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_w, bytes));

  lab4::Workspace ws;
  const std::size_t pairs = lab4::workspace_pairs(n);
  CUDA_CHECK(cudaMalloc(&ws.a, pairs * sizeof(Pair)));
  CUDA_CHECK(cudaMalloc(&ws.b, pairs * sizeof(Pair)));
  CUDA_CHECK(cudaMalloc(&ws.result, sizeof(Pair)));
  ws.cub_bytes = lab4::cub_temp_bytes(n);
  CUDA_CHECK(cudaMalloc(&ws.cub_temp, std::max<std::size_t>(ws.cub_bytes, 1)));

  const std::size_t flush_bytes = 2 * static_cast<std::size_t>(dev.l2_bytes);
  void* flush = nullptr;
  if (flush_bytes > 0) CUDA_CHECK(cudaMalloc(&flush, flush_bytes));

  double h2d_ms = 0.0;
  {
    Row row;
    time_gpu(
        [&] {
          CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_w, h_w, bytes, cudaMemcpyHostToDevice));
        },
        opt.repeat, row, flush, flush_bytes);
    h2d_ms = row.best_ms;
  }

  auto run = [&](Row row, auto&& fn) {
    CUDA_CHECK(cudaMemset(ws.result, 0xFF, sizeof(Pair)));
    time_gpu(fn, opt.repeat, row, flush, flush_bytes);
    CUDA_CHECK(cudaMemcpy(&row.value, ws.result, sizeof(Pair), cudaMemcpyDeviceToHost));
    rows.push_back(row);
  };

  const struct {
    Tree tree;
    const char* name;
  } trees[] = {{Tree::interleaved, "G1 interleaved, "}, {Tree::sequential, "G2 sequential, "}};
  for (const auto& t : trees) {
    Row row;
    row.title = t.name + std::to_string(opt.tree_block);
    row.check = Check::bitwise;
    row.expected = lab4::emulate_tree(t.tree, h_x, h_w, n, opt.tree_block);
    run(row, [&] { lab4::launch_tree(t.tree, d_x, d_w, n, opt.tree_block, ws); });
  }

  for (const int block : opt.blocks) {
    const int cap = dev.sm_count * std::max(1, dev.threads_per_sm / block);
    const int grid = lab4::grid_stride_blocks(n, block, cap);
    Row row;
    row.title = "G3 grid-stride+shfl, " + std::to_string(block) + "×" + std::to_string(grid);
    row.check = Check::bitwise;
    row.expected = lab4::emulate_grid_stride(h_x, h_w, n, block, grid);
    run(row, [&] { lab4::launch_grid_stride(d_x, d_w, n, block, grid, ws); });
  }

  {
    Row row;
    row.title = "CUB DeviceReduce";
    row.check = Check::tolerance;
    run(row, [&] { lab4::launch_cub(d_x, d_w, n, ws); });
  }

  const double cpu_ms = rows.front().best_ms;
  const double data_bytes = 2.0 * bytes;
  const std::size_t w_title = 32;
  std::printf("│ %s │ %s │ %s │ %s │ %s │ %s │ %s │\n", pad_right("Версия", w_title).c_str(),
              pad_left("Лучшее, мс", 10).c_str(), pad_left("Среднее, мс", 11).c_str(),
              pad_left("ГБ/с", 7).c_str(), pad_left("vs CPU", 7).c_str(),
              pad_left("Отн. ош.", 8).c_str(), pad_left("Сверка", 12).c_str());
  std::printf("├%s┼%s┼%s┼%s┼%s┼%s┼%s┤\n", rule(w_title + 2).c_str(), rule(12).c_str(),
              rule(13).c_str(), rule(9).c_str(), rule(9).c_str(), rule(10).c_str(),
              rule(14).c_str());

  bool all_ok = true;
  const Row* best_gpu = nullptr;
  for (const Row& row : rows) {
    const double err = rel_error(row.value, exact);
    std::string verdict = "—";
    if (row.check == Check::bitwise) {
      const bool ok = same_bits(row.value, row.expected);
      verdict = ok ? "побитово" : "≠ эмуляции";
      all_ok = all_ok && ok;
    } else if (row.check == Check::tolerance) {
      const bool ok = err <= kCubTolerance;
      verdict = ok ? "≤ 1e-5" : "> 1e-5";
      all_ok = all_ok && ok;
    }
    std::printf("│ %s │ %s │ %s │ %s │ %s │ %s │ %s │\n", pad_right(row.title, w_title).c_str(),
                pad_left(fmt("%.4f", row.best_ms), 10).c_str(),
                pad_left(fmt("%.4f", row.mean_ms), 11).c_str(),
                pad_left(fmt("%.1f", data_bytes / (row.best_ms * 1e6)), 7).c_str(),
                pad_left(fmt("%.1f×", cpu_ms / row.best_ms), 7).c_str(),
                pad_left(fmt("%.1e", err), 8).c_str(), pad_left(verdict, 12).c_str());
    if (!row.is_cpu && (best_gpu == nullptr || row.best_ms < best_gpu->best_ms)) best_gpu = &row;
  }
  std::printf("\n");

  std::printf("x̄_w точно (double):         %.9f\n", exact);
  std::printf("x̄_w CPU, float подряд:      %.9f   Σwx = %.6e, Σw = %.6e\n",
              mean_of(rows.front().value), rows.front().value.wx, rows.front().value.w);
  if (best_gpu != nullptr) {
    std::printf("x̄_w лучшей GPU-версии:      %.9f   Σwx = %.6e, Σw = %.6e\n",
                mean_of(best_gpu->value), best_gpu->value.wx, best_gpu->value.w);
  }
  std::printf("\nКопирование x и w CPU→GPU (%.1f МБ, pinned): %.4f мс, %.1f ГБ/с\n",
              data_bytes / (1024.0 * 1024.0), h2d_ms, data_bytes / (h2d_ms * 1e6));
  if (best_gpu != nullptr) {
    std::printf("Лучшая GPU-версия: %s, с копированием %.4f мс (%.2f× к CPU)\n",
                best_gpu->title.c_str(), best_gpu->best_ms + h2d_ms,
                cpu_ms / (best_gpu->best_ms + h2d_ms));
  }

  if (flush != nullptr) CUDA_CHECK(cudaFree(flush));
  CUDA_CHECK(cudaFree(ws.cub_temp));
  CUDA_CHECK(cudaFree(ws.result));
  CUDA_CHECK(cudaFree(ws.b));
  CUDA_CHECK(cudaFree(ws.a));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFreeHost(h_w));
  CUDA_CHECK(cudaFreeHost(h_x));

  if (!all_ok) {
    std::fprintf(stderr, "\nРезультаты GPU не совпали с эталоном.\n");
    return EXIT_FAILURE;
  }
  std::printf("Все версии G1–G3 совпали со своими CPU-эмуляциями побитово, CUB — в допуске.\n");
  return EXIT_SUCCESS;
}
