#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "image.hpp"
#include "kernels.cuh"

namespace {

using lab3::JuliaParams;

struct Options {
  float cx = -0.8f;
  float cy = 0.156f;
  float x_min = -1.6f;
  float x_max = 1.6f;
  float y_min = -0.9f;
  float y_max = 0.9f;
  int width = 4096;
  int height = 2304;
  int max_iter = 1000;
  int repeat = 10;
  bool cpu = true;
  std::string out;
  std::vector<dim3> blocks = {dim3(8, 8), dim3(16, 8), dim3(16, 16), dim3(32, 4),
                              dim3(32, 8), dim3(32, 16), dim3(32, 32)};
};

struct Row {
  std::string key;
  std::string title;
  double best_ms = 0.0;
  double mean_ms = 0.0;
  std::size_t mismatches = 0;
  bool is_reference = false;
};

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
      "Лабораторная работа №3, задание 10: множество Жюлиа на CUDA.\n\n"
      "Использование: %s [ключи]\n\n"
      "  --c CX CY                  параметр c = CX + i·CY, CX, CY ∈ [-1.5, 1.5]\n"
      "                             (по умолчанию -0.8 0.156)\n"
      "  --area XMIN XMAX YMIN YMAX границы области (по умолчанию -1.6 1.6 -0.9 0.9)\n"
      "  --size W H                 размер сетки, W, H ∈ [100, 4096] (по умолчанию 4096 2304)\n"
      "  --iter NMAX                Nmax ∈ [50, 1000] (по умолчанию 1000)\n"
      "  --repeat R                 число замеров каждого ядра (по умолчанию 10)\n"
      "  --block BXxBY              проверить только один размер блока, например 32x8\n"
      "  --no-cpu                   не считать эталон на CPU (сверка с G1)\n"
      "  --out FILE.ppm             сохранить картинку\n"
      "  -h, --help                 эта справка\n",
      program);
}

float parse_float(const std::string& text, const char* name) {
  std::size_t used = 0;
  float value = 0.0f;
  try {
    value = std::stof(text, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string("неверное число для ") + name + ": " + text);
  }
  return value;
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

dim3 parse_block(const std::string& text) {
  const std::size_t sep = text.find('x');
  if (sep == std::string::npos) throw std::invalid_argument("блок задаётся как BXxBY: " + text);
  const int bx = parse_int(text.substr(0, sep), "--block");
  const int by = parse_int(text.substr(sep + 1), "--block");
  if (bx < 1 || by < 1 || bx * by > 1024) {
    throw std::invalid_argument("размер блока должен быть от 1 до 1024 потоков: " + text);
  }
  return dim3(bx, by);
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
    } else if (arg == "--c") {
      need(i, 2, arg);
      opt.cx = parse_float(argv[++i], "cx");
      opt.cy = parse_float(argv[++i], "cy");
    } else if (arg == "--area") {
      need(i, 4, arg);
      opt.x_min = parse_float(argv[++i], "xmin");
      opt.x_max = parse_float(argv[++i], "xmax");
      opt.y_min = parse_float(argv[++i], "ymin");
      opt.y_max = parse_float(argv[++i], "ymax");
    } else if (arg == "--size") {
      need(i, 2, arg);
      opt.width = parse_int(argv[++i], "W");
      opt.height = parse_int(argv[++i], "H");
    } else if (arg == "--iter") {
      need(i, 1, arg);
      opt.max_iter = parse_int(argv[++i], "Nmax");
    } else if (arg == "--repeat") {
      need(i, 1, arg);
      opt.repeat = parse_int(argv[++i], "--repeat");
    } else if (arg == "--block") {
      need(i, 1, arg);
      opt.blocks = {parse_block(argv[++i])};
    } else if (arg == "--no-cpu") {
      opt.cpu = false;
    } else if (arg == "--out") {
      need(i, 1, arg);
      opt.out = argv[++i];
    } else {
      throw std::invalid_argument("неизвестный ключ: " + arg);
    }
  }

  if (opt.cx < -1.5f || opt.cx > 1.5f || opt.cy < -1.5f || opt.cy > 1.5f) {
    throw std::invalid_argument("cx, cy должны лежать в [-1.5, 1.5]");
  }
  if (opt.width < 100 || opt.width > 4096 || opt.height < 100 || opt.height > 4096) {
    throw std::invalid_argument("W, H должны лежать в [100, 4096]");
  }
  if (opt.max_iter < 50 || opt.max_iter > 1000) {
    throw std::invalid_argument("Nmax должно лежать в [50, 1000]");
  }
  if (!(opt.x_min < opt.x_max) || !(opt.y_min < opt.y_max)) {
    throw std::invalid_argument("нужно xmin < xmax и ymin < ymax");
  }
  if (opt.repeat < 1) throw std::invalid_argument("--repeat должно быть не меньше 1");
  return opt;
}

JuliaParams make_params(const Options& opt) {
  JuliaParams p;
  p.cx = opt.cx;
  p.cy = opt.cy;
  p.x_min = opt.x_min;
  p.y_max = opt.y_max;
  p.dx = (opt.x_max - opt.x_min) / static_cast<float>(opt.width - 1);
  p.dy = (opt.y_max - opt.y_min) / static_cast<float>(opt.height - 1);
  p.width = opt.width;
  p.height = opt.height;
  p.max_iter = opt.max_iter;
  return p;
}

template <typename Fn>
void time_gpu(Fn&& fn, int repeat, Row& row) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  fn();
  CUDA_CHECK(cudaDeviceSynchronize());

  double best = std::numeric_limits<double>::infinity();
  double sum = 0.0;
  for (int r = 0; r < repeat; ++r) {
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

std::size_t count_mismatches(const std::uint16_t* a, const std::uint16_t* b, std::size_t n) {
  std::size_t bad = 0;
  for (std::size_t i = 0; i < n; ++i) bad += a[i] != b[i];
  return bad;
}

void print_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime = 0;
  int driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  std::printf("GPU:        %s, CC %d.%d, %d SM, %.1f ГБ\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("CUDA:       runtime %d.%d, driver %d.%d\n", runtime / 1000, runtime % 1000 / 10,
              driver / 1000, driver % 1000 / 10);
}

}

int main(int argc, char** argv) {
  Options opt;
  try {
    opt = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Ошибка: %s\nПодсказка: %s --help\n", e.what(), argv[0]);
    return EXIT_FAILURE;
  }

  const JuliaParams p = make_params(opt);
  const std::size_t pixels = static_cast<std::size_t>(p.width) * p.height;
  const std::size_t out_bytes = pixels * sizeof(std::uint16_t);

  print_device();
  std::printf("Задача:     c = %g %+gi, область [%g, %g] × [%g, %g]\n", opt.cx, opt.cy,
              opt.x_min, opt.x_max, opt.y_min, opt.y_max);
  std::printf("Сетка:      %d × %d = %zu точек float2, Nmax = %d, замеров ядра: %d\n\n",
              p.width, p.height, pixels, p.max_iter, opt.repeat);

  float2* d_grid = nullptr;
  std::uint16_t* d_out = nullptr;
  std::uint16_t* h_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_grid, pixels * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, out_bytes));

  std::vector<Row> rows;
  std::vector<std::uint16_t> reference(pixels);
  bool have_reference = false;

  if (opt.cpu) {
    Row row;
    row.key = "cpu";
    row.title = "CPU, 1 поток (эталон)";
    row.is_reference = true;
    const auto begin = std::chrono::high_resolution_clock::now();
    lab3::julia_reference(p, reference.data());
    const auto end = std::chrono::high_resolution_clock::now();
    row.best_ms = row.mean_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    rows.push_back(row);
    have_reference = true;
  }

  auto check = [&](Row& row) {
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    if (!have_reference) {
      std::copy(h_out, h_out + pixels, reference.begin());
      have_reference = true;
      row.is_reference = true;
      return;
    }
    row.mismatches = count_mismatches(h_out, reference.data(), pixels);
  };

  const dim3 two_pass_block(16, 16);
  {
    Row row;
    row.key = "g1";
    row.title = "G1 сетка float2 + итерации, 16×16";
    time_gpu(
        [&] {
          lab3::launch_make_grid(d_grid, p, two_pass_block);
          lab3::launch_julia_from_grid(d_grid, d_out, p, two_pass_block);
        },
        opt.repeat, row);
    check(row);
    rows.push_back(row);
  }

  for (const dim3& block : opt.blocks) {
    Row row;
    row.key = "g2";
    row.title = "G2 слитное ядро, " + std::to_string(block.x) + "×" + std::to_string(block.y);
    CUDA_CHECK(cudaMemset(d_out, 0xFF, out_bytes));
    time_gpu([&] { lab3::launch_julia_fused(d_out, p, block); }, opt.repeat, row);
    check(row);
    rows.push_back(row);
  }

  double copy_ms = 0.0;
  {
    Row row;
    time_gpu([&] { CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost)); },
             opt.repeat, row);
    copy_ms = row.best_ms;
  }

  std::uint64_t total_iter = 0;
  std::size_t inside = 0;
  for (const std::uint16_t n : reference) {
    total_iter += n;
    inside += n >= p.max_iter;
  }

  const double cpu_ms = opt.cpu ? rows.front().best_ms : 0.0;
  const std::size_t w_title = 38;
  std::printf("│ %s │ %s │ %s │ %s │ %s │ %s │\n", pad_right("Версия", w_title).c_str(),
              pad_left("Лучшее, мс", 11).c_str(), pad_left("Среднее, мс", 11).c_str(),
              pad_left("Гитер/с", 8).c_str(), pad_left("vs CPU", 8).c_str(),
              pad_left("Сверка", 12).c_str());
  std::printf("├%s┼%s┼%s┼%s┼%s┼%s┤\n", rule(w_title + 2).c_str(), rule(13).c_str(),
              rule(13).c_str(), rule(10).c_str(), rule(10).c_str(), rule(14).c_str());

  bool all_ok = true;
  const Row* best_gpu = nullptr;
  for (const Row& row : rows) {
    const std::string verdict =
        row.is_reference ? "эталон"
        : row.mismatches == 0 ? "побитово"
                              : std::to_string(row.mismatches) + " разн.";
    all_ok = all_ok && row.mismatches == 0;
    const double giter = static_cast<double>(total_iter) / (row.best_ms * 1e6);
    const std::string speedup = opt.cpu ? fmt("%.1f×", cpu_ms / row.best_ms) : "—";
    std::printf("│ %s │ %s │ %s │ %s │ %s │ %s │\n", pad_right(row.title, w_title).c_str(),
                pad_left(fmt("%.3f", row.best_ms), 11).c_str(),
                pad_left(fmt("%.3f", row.mean_ms), 11).c_str(),
                pad_left(fmt("%.2f", giter), 8).c_str(), pad_left(speedup, 8).c_str(),
                pad_left(verdict, 12).c_str());
    if (row.key != "cpu" && (best_gpu == nullptr || row.best_ms < best_gpu->best_ms)) {
      best_gpu = &row;
    }
  }
  std::printf("\n");

  std::printf("Копирование результата GPU→CPU (%.1f МБ, pinned): %.3f мс\n",
              out_bytes / (1024.0 * 1024.0), copy_ms);
  if (best_gpu != nullptr) {
    std::printf("Лучшая GPU-версия: %s, с копированием %.3f мс", best_gpu->title.c_str(),
                best_gpu->best_ms + copy_ms);
    if (opt.cpu) std::printf(" (%.1f× к CPU)", cpu_ms / (best_gpu->best_ms + copy_ms));
    std::printf("\n");
  }
  std::printf("Точек в множестве (n = Nmax): %zu из %zu (%.2f%%), всего итераций: %llu\n",
              inside, pixels, 100.0 * inside / pixels,
              static_cast<unsigned long long>(total_iter));

  if (!opt.out.empty()) {
    if (lab3::write_ppm(opt.out, reference.data(), p.width, p.height, p.max_iter)) {
      std::printf("Картинка:   %s\n", opt.out.c_str());
    } else {
      std::fprintf(stderr, "Не удалось записать %s\n", opt.out.c_str());
    }
  }

  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_grid));

  if (!all_ok) {
    std::fprintf(stderr, "\nРезультаты GPU не совпали с эталоном.\n");
    return EXIT_FAILURE;
  }
  std::printf("Все версии совпали с эталоном побитово.\n");
  return EXIT_SUCCESS;
}
