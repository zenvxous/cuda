#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cache_info.hpp"
#include "cache_probe.hpp"
#include "matrix.hpp"
#include "multiply.hpp"
#include "timer.hpp"

#ifndef LAB2_CXXFLAGS
#define LAB2_CXXFLAGS "(не заданы)"
#endif

namespace {

using lab2::CacheInfo;
using lab2::CacheLevel;
using lab2::Matrix;

struct Options {
  int rows = 2560;
  int inner = 2560;
  int cols = 2560;
  int repeat = 1;
  int tile = 0;
  std::size_t llc = 0;
  std::size_t l2 = 0;
  std::size_t l1 = 0;
  std::size_t line = 0;
  bool with_naive = false;
  bool diag = false;
  bool probe = false;
  bool info = false;
  bool csv = false;
  std::uint64_t seed = 20240501;
  std::vector<std::string> only;
};

struct Variant {
  std::string key;
  std::string title;
  std::string note;
  bool executed = false;
  double best = 0.0;
  double mean = 0.0;
  double ticks = 0.0;
  double passes_a = 0.0;
  double passes_b = 0.0;
  double passes_c = 0.0;
  Matrix* result = nullptr;
  std::function<void()> run;
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

std::string repeat_utf8(const std::string& unit, std::size_t times) {
  std::string result;
  for (std::size_t i = 0; i < times; ++i) result += unit;
  return result;
}

std::string format_number(double value, int precision) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
  return buffer;
}

std::string scaled(double value, const char* const* units, int max_unit, double step) {
  int unit = 0;
  while (value >= step && unit < max_unit) {
    value /= step;
    ++unit;
  }
  return format_number(value, unit == 0 ? 0 : 2) + units[unit];
}

std::string format_bytes(double bytes) {
  static const char* const units[] = {" Б", " КБ", " МБ", " ГБ", " ТБ"};
  return scaled(bytes, units, 4, 1024.0);
}

std::string format_ticks(double ticks) {
  static const char* const units[] = {"", " тыс.", " млн", " млрд"};
  return scaled(ticks, units, 3, 1000.0);
}

void print_field(const std::string& name, const std::string& value, std::size_t width = 24) {
  std::string dotted = name + " ";
  while (visual_width(dotted) < width) dotted += ".";
  std::cout << "  " << dotted << " " << value << "\n";
}

void print_usage(const char* program) {
  std::cout <<
      "Лабораторная работа №2 — оптимизация умножения матриц double под кэш (NEON).\n\n"
      "Использование:\n"
      "  " << program << " [параметры]\n\n"
      "Параметры:\n"
      "  -s, --size N|M,K,N   размеры матриц: A[M×K] · B[K×N] = C[M×N]. По умолчанию 2560.\n"
      "  -r, --repeat K       прогонов каждой версии. По умолчанию 1.\n"
      "      --only список    какие версии считать: naive,c1,ikj,reg,blk0,c2,c3\n"
      "                       (c1 считается всегда — она эталон для сравнения).\n"
      "      --with-naive     добавить опорную скалярную версию (самая долгая).\n"
      "      --diag           добавить диагностические версии C1′ и C2′ (разделяют эффекты).\n"
      "      --tile T         принудительный размер тайла для C2 вместо расчётного.\n"
      "      --llc РАЗМЕР     переопределить размер LLC (суффиксы K/M, напр. 8M).\n"
      "      --l2 РАЗМЕР      переопределить размер L2.\n"
      "      --l1 РАЗМЕР      переопределить размер L1d.\n"
      "      --line БАЙТ      переопределить размер линии кэша.\n"
      "      --probe          замерить иерархию кэша и выйти.\n"
      "      --info           показать кэши и рассчитанные размеры блоков и выйти.\n"
      "      --seed N         зерно генератора данных. По умолчанию 20240501.\n"
      "      --csv            машиночитаемый вывод.\n"
      "  -h, --help           эта справка.\n\n"
      "Примеры:\n"
      "  " << program << "                        прогон с настройками по умолчанию\n"
      "  " << program << " --size 512 --with-naive  малый размер со всеми версиями\n"
      "  " << program << " --llc 8M --l2 1M --l1 48K  цифры типовой x86-машины\n"
      "  " << program << " --probe                  профиль иерархии кэша\n";
}

bool parse_positive(const char* text, long long& out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) return false;
  out = value;
  return true;
}

bool parse_size_value(const char* text, std::size_t& out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || value <= 0) return false;
  long long multiplier = 1;
  if (*end == 'K' || *end == 'k') multiplier = 1024, ++end;
  else if (*end == 'M' || *end == 'm') multiplier = 1024 * 1024, ++end;
  if (*end != '\0') return false;
  out = static_cast<std::size_t>(value * multiplier);
  return true;
}

bool parse_dims(const char* text, Options& options) {
  std::vector<std::string> parts(1);
  for (const char* p = text; *p != '\0'; ++p) {
    if (*p == ',' || *p == 'x') parts.emplace_back();
    else parts.back().push_back(*p);
  }
  if (parts.size() != 1 && parts.size() != 3) return false;

  long long values[3];
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (!parse_positive(parts[i].c_str(), values[i]) || values[i] > 100000) return false;
  }
  if (parts.size() == 1) values[1] = values[2] = values[0];

  options.rows = static_cast<int>(values[0]);
  options.inner = static_cast<int>(values[1]);
  options.cols = static_cast<int>(values[2]);
  return true;
}

void split_list(const char* text, std::vector<std::string>& out) {
  std::string current;
  for (const char* p = text;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!current.empty()) out.push_back(current);
      current.clear();
      if (*p == '\0') break;
    } else {
      current.push_back(*p);
    }
  }
}

bool parse_options(int argc, char** argv, Options& options, int& exit_code) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_next = (i + 1 < argc);
    long long value = 0;
    std::size_t size_value = 0;

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      exit_code = 0;
      return false;
    }
    if (arg == "--with-naive") { options.with_naive = true; continue; }
    if (arg == "--diag") { options.diag = true; continue; }
    if (arg == "--probe") { options.probe = true; continue; }
    if (arg == "--info") { options.info = true; continue; }
    if (arg == "--csv") { options.csv = true; continue; }

    if (arg == "-s" || arg == "--size") {
      if (has_next && parse_dims(argv[++i], options)) continue;
      std::cerr << "Ошибка: --size ожидает N или M,K,N (положительные числа).\n";
    } else if (arg == "-r" || arg == "--repeat") {
      if (has_next && parse_positive(argv[++i], value) && value <= 1000) {
        options.repeat = static_cast<int>(value);
        continue;
      }
      std::cerr << "Ошибка: --repeat ожидает число от 1 до 1000.\n";
    } else if (arg == "--tile") {
      if (has_next && parse_positive(argv[++i], value)) {
        options.tile = static_cast<int>(value);
        continue;
      }
      std::cerr << "Ошибка: --tile ожидает положительное число.\n";
    } else if (arg == "--seed") {
      if (has_next && parse_positive(argv[++i], value)) {
        options.seed = static_cast<std::uint64_t>(value);
        continue;
      }
      std::cerr << "Ошибка: --seed ожидает положительное число.\n";
    } else if (arg == "--llc" || arg == "--l3") {
      if (has_next && parse_size_value(argv[++i], size_value)) { options.llc = size_value; continue; }
      std::cerr << "Ошибка: " << arg << " ожидает размер, например 8M.\n";
    } else if (arg == "--l2") {
      if (has_next && parse_size_value(argv[++i], size_value)) { options.l2 = size_value; continue; }
      std::cerr << "Ошибка: --l2 ожидает размер, например 1M.\n";
    } else if (arg == "--l1") {
      if (has_next && parse_size_value(argv[++i], size_value)) { options.l1 = size_value; continue; }
      std::cerr << "Ошибка: --l1 ожидает размер, например 48K.\n";
    } else if (arg == "--line") {
      if (has_next && parse_size_value(argv[++i], size_value)) { options.line = size_value; continue; }
      std::cerr << "Ошибка: --line ожидает размер в байтах.\n";
    } else if (arg == "--only") {
      if (has_next) { split_list(argv[++i], options.only); continue; }
      std::cerr << "Ошибка: --only ожидает список, например c1,c2,c3.\n";
    } else {
      std::cerr << "Ошибка: неизвестный параметр \"" << arg << "\".\n"
                << "Подсказка: " << argv[0] << " --help\n";
    }
    exit_code = 2;
    return false;
  }
  return true;
}

void print_table(const std::vector<std::size_t>& widths, const std::vector<std::string>& header,
                 const std::vector<std::vector<std::string>>& rows) {
  const std::size_t columns = widths.size();
  auto rule = [&](const char* left, const char* mid, const char* right) {
    std::cout << left;
    for (std::size_t i = 0; i < columns; ++i) {
      std::cout << repeat_utf8("─", widths[i] + 2) << (i + 1 == columns ? right : mid);
    }
    std::cout << "\n";
  };
  auto line = [&](const std::vector<std::string>& cells) {
    std::cout << "│ " << pad_right(cells[0], widths[0]);
    for (std::size_t i = 1; i < columns; ++i) std::cout << " │ " << pad_left(cells[i], widths[i]);
    std::cout << " │\n";
  };

  rule("┌", "┬", "┐");
  line(header);
  rule("├", "┼", "┤");
  for (const auto& row : rows) line(row);
  rule("└", "┴", "┘");
}

void run_probe(const CacheInfo& info) {
  std::cout << "\nПрофиль иерархии кэша (pointer chasing, шаг " << info.line_size
            << " Б, случайный порядок)\n"
            << "Часы: std::chrono::high_resolution_clock.\n\n";

  const std::vector<lab2::ProbePoint> points =
      lab2::probe_cache(info.line_size, 512u * 1024 * 1024);

  double max_latency = 0.0;
  for (const auto& point : points) max_latency = std::max(max_latency, point.ns_per_access);

  std::cout << "  Рабочее множество   нс/доступ\n";
  double previous = 0.0;
  for (const auto& point : points) {
    const std::size_t bars =
        static_cast<std::size_t>(point.ns_per_access / max_latency * 46.0 + 0.5);
    std::string mark;
    if (previous > 0.0 && point.ns_per_access > previous * 1.4) mark = "  ← ступень";
    if (point.bytes == info.l1d) mark += "  [= L1d]";
    if (point.bytes == info.l2) mark += "  [= L2]";
    std::cout << "  " << pad_left(format_bytes(static_cast<double>(point.bytes)), 12) << "  "
              << pad_left(format_number(point.ns_per_access, 2), 8) << "  "
              << repeat_utf8("█", bars) << mark << "\n";
    previous = point.ns_per_access;
  }

  std::cout << "\nСтупени на графике — границы уровней кэша. Размер блока в задании выбирается\n"
               "так, чтобы три блока (A, B, C) оставались левее ступени LLC.\n\n";
}

}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);

  Options options;
  int exit_code = 0;
  if (!parse_options(argc, argv, options, exit_code)) return exit_code;

  CacheInfo caches = lab2::detect_caches();
  lab2::apply_overrides(caches, options.l1, options.l2, options.llc, options.line);

  if (options.probe) {
    run_probe(caches);
    return 0;
  }

  const bool quiet = options.csv;
  const int m = options.rows;
  const int k_size = options.inner;
  const int n = options.cols;

  const double gflop = 2.0 * m * k_size * n / 1e9;
  const std::size_t budget = lab2::block_budget_bytes(caches.llc);
  const int tile = options.tile > 0 ? options.tile
                                    : lab2::tile_side(caches.llc, caches.line_size);
  const std::vector<CacheLevel> levels = lab2::blocking_levels(caches);
  std::vector<int> multi_tiles;
  for (const CacheLevel& level : levels) {
    multi_tiles.push_back(lab2::tile_side(level.size, caches.line_size));
  }

  auto wanted = [&](const std::string& key) {
    if (key == "c1") return true;
    if (options.only.empty()) {
      if (key == "naive") return options.with_naive;
      if (key == "ikj" || key == "reg" || key == "blk0") return options.diag;
      return true;
    }
    return std::find(options.only.begin(), options.only.end(), key) != options.only.end();
  };

  try {
    Matrix a(m, k_size);
    Matrix b(k_size, n);
    lab2::fill_random(a, options.seed);
    lab2::fill_random(b, options.seed * 6364136223846793005ULL + 1442695040888963407ULL);

    std::vector<Variant> variants;
    std::vector<std::unique_ptr<Matrix>> results;

    auto add = [&](const std::string& key, const std::string& title, const std::string& note,
                   double pa, double pb, double pc,
                   const std::function<void(Matrix&)>& body) {
      if (!wanted(key)) return;
      results.push_back(std::make_unique<Matrix>(m, n));
      Matrix* const target = results.back().get();
      Variant variant;
      variant.key = key;
      variant.title = title;
      variant.note = note;
      variant.passes_a = pa;
      variant.passes_b = pb;
      variant.passes_c = pc;
      variant.result = target;
      variant.run = [body, target] { body(*target); };
      variants.push_back(variant);
    };

    const double passes_naive_b = static_cast<double>(m);
    const double passes_c1_b = std::ceil(m / 2.0);
    const double passes_tile_a = std::ceil(static_cast<double>(n) / tile);
    const double passes_tile_b = std::ceil(static_cast<double>(m) / tile);
    const double passes_tile_c = 2.0 * std::ceil(static_cast<double>(k_size) / tile);
    const int outer_tile = multi_tiles.empty() ? tile : multi_tiles.front();
    const double passes_multi_a = std::ceil(static_cast<double>(n) / outer_tile);
    const double passes_multi_b = std::ceil(static_cast<double>(m) / outer_tile);
    const double passes_multi_c = 2.0 * std::ceil(static_cast<double>(k_size) / outer_tile);

    add("naive", "C0  скалярная i-j-k (опорная)", "без векторизации", 1.0, passes_naive_b,
        1.0, [&](Matrix& out) { lab2::multiply_naive(a, b, out); });
    add("c1", "C1  векторизованная, без оптим. кэша", "порядок i-j-k, как в лаб.1", 1.0,
        passes_naive_b, 1.0,
        [&](Matrix& out) { lab2::multiply_vec_ijk(a, b, out); });
    add("ikj", "C1a перестановка циклов на i-k-j", "диагностика: шаг 1", 1.0, passes_c1_b,
        2.0, [&](Matrix& out) { lab2::multiply_vec(a, b, out); });
    add("reg", "C1′ регистровая блокировка, без кэша", "диагностика", 1.0,
        std::ceil(static_cast<double>(m) / 4.0), 2.0,
        [&](Matrix& out) { lab2::multiply_reg(a, b, out); });
    add("blk0", "C2′ кэш-блокировка с ядром C1a", "диагностика", passes_tile_a,
        passes_tile_b, passes_tile_c,
        [&](Matrix& out) { lab2::multiply_block_plain(a, b, out, tile); });
    add("c2", "C2  блочная по заданию (LLC)", "blocksize = LLC/3 · 0,9", passes_tile_a,
        passes_tile_b, passes_tile_c,
        [&](Matrix& out) { lab2::multiply_block(a, b, out, tile); });
    add("c3", "C3  по всей иерархии кэша", "вложенный тайлинг", passes_multi_a,
        passes_multi_b, passes_multi_c,
        [&](Matrix& out) { lab2::multiply_block_multi(a, b, out, multi_tiles); });

    if (!quiet) {
      std::cout << "\n╔" << repeat_utf8("═", 76) << "╗\n║ "
                << pad_right("ЛАБОРАТОРНАЯ РАБОТА №2 — оптимизация умножения матриц под кэш", 75)
                << "║\n║ " << pad_right("Тип double, ручная векторизация NEON, плотные матрицы", 75)
                << "║\n╚" << repeat_utf8("═", 76) << "╝\n\n";

      std::cout << "Машина\n";
      print_field("Архитектура", "arm64 (AArch64)");
      print_field("Ручной SIMD", "NEON (регистры v, 128 бит = 2 × double)");
      print_field("Компилятор", __VERSION__);
      print_field("Флаги", LAB2_CXXFLAGS);
      print_field("Линия кэша", std::to_string(caches.line_size) + " байт" +
                                   (caches.line_size % 64 == 0
                                        ? " (кратно 64 — требование задания выполнено)"
                                        : " (НЕ кратно 64)"));
      print_field("L1d", format_bytes(static_cast<double>(caches.l1d)));
      print_field("L2", format_bytes(static_cast<double>(caches.l2)));
      print_field("L3", caches.l3 != 0 ? format_bytes(static_cast<double>(caches.l3))
                                       : "отсутствует (Apple Silicon)");
      print_field("LLC для формулы", format_bytes(static_cast<double>(caches.llc)) + " — " +
                                         caches.llc_origin);
      print_field("Счётчик тактов", "cntvct_el0, " +
                                        format_number(lab2::tick_counter_hz() / 1e6, 2) + " МГц");
      print_field("Часы", "std::chrono::high_resolution_clock");

      std::cout << "\nРазмеры блоков (формула задания: blocksize = LLC / 3 · 0,9)\n";
      print_field("Бюджет одного блока", format_bytes(static_cast<double>(budget)) +
                                             " (в кэше держим блоки A, B и C)");
      print_field("Сторона тайла T", std::to_string(tile) + " элементов" +
                                         (options.tile > 0 ? " (задана флагом --tile)" : ""));
      print_field("Ширина блока A и B", std::to_string(tile * sizeof(lab2::Elem)) + " байт — " +
                                            ((tile * sizeof(lab2::Elem)) % 64 == 0
                                                 ? "кратно 64: ДА"
                                                 : "кратно 64: НЕТ"));
      print_field("Три блока занимают", format_bytes(3.0 * tile * tile * sizeof(lab2::Elem)) +
                                            " из " + format_bytes(static_cast<double>(caches.llc)));
      print_field("Блоков в матрице C",
                  std::to_string(static_cast<int>(std::ceil(static_cast<double>(m) / tile))) +
                      " × " +
                      std::to_string(static_cast<int>(std::ceil(static_cast<double>(n) / tile))));
      {
        std::string chain;
        for (std::size_t i = 0; i < levels.size(); ++i) {
          if (i != 0) chain += " → ";
          chain += levels[i].origin + " " +
                   format_bytes(static_cast<double>(levels[i].size)) + ", T=" +
                   std::to_string(multi_tiles[i]);
        }
        print_field("Уровни для C3", chain + "  (" + std::to_string(levels.size()) + " уровня)");
      }

      if (options.info) {
        std::cout << "\n";
        return 0;
      }

      std::cout << "\nЗадача\n";
      print_field("Матрицы", "A[" + std::to_string(m) + "×" + std::to_string(k_size) + "] · B[" +
                                 std::to_string(k_size) + "×" + std::to_string(n) + "] = C[" +
                                 std::to_string(m) + "×" + std::to_string(n) + "]");
      print_field("mod(size, 64)",
                  std::to_string(m % 64) + " / " + std::to_string(k_size % 64) + " / " +
                      std::to_string(n % 64) +
                      ((m % 64 == 0 && k_size % 64 == 0 && n % 64 == 0)
                           ? " — требование задания выполнено"
                           : " — размеры не кратны 64, крайние тайлы неполные"));
      print_field("Шаг строки (stride)", std::to_string(a.stride()) + " элементов = " +
                                             std::to_string(a.stride() * sizeof(lab2::Elem)) +
                                             " байт");
      print_field("Память", format_bytes(static_cast<double>(a.bytes() + b.bytes()) +
                                         static_cast<double>(results.size()) *
                                             (m * static_cast<double>(n) * sizeof(lab2::Elem))));
      print_field("Объём вычислений", format_number(gflop, 2) + " GFLOP");
      print_field("Прогонов на версию", std::to_string(options.repeat));

      std::cout << "\nЗамеры\n";
    }

    for (Variant& variant : variants) {
      double total = 0.0;
      for (int run = 0; run < options.repeat; ++run) {
        if (!quiet) {
          std::cout << "  " << pad_right(variant.title, 38) << " прогон " << (run + 1) << "/"
                    << options.repeat << " … " << std::flush;
        }
        const lab2::Measurement measurement = lab2::measure(variant.run);
        total += measurement.seconds;
        if (run == 0 || measurement.seconds < variant.best) {
          variant.best = measurement.seconds;
          variant.ticks = static_cast<double>(measurement.ticks);
        }
        if (!quiet) std::cout << format_number(measurement.seconds, 3) << " с\n";
      }
      variant.mean = total / options.repeat;
      variant.executed = true;
    }

    const Variant* reference = nullptr;
    for (const Variant& variant : variants) {
      if (variant.key == "c1") reference = &variant;
    }

    std::vector<lab2::CompareResult> checks(variants.size());
    bool all_match = true;
    for (std::size_t i = 0; i < variants.size(); ++i) {
      checks[i] = lab2::compare(*reference->result, *variants[i].result);
      if (!checks[i].bit_exact) all_match = false;
    }

    const double baseline = reference->best;

    if (options.csv) {
      std::cout << "version,best_seconds,mean_seconds,ticks,gflops,speedup,bit_exact\n";
      for (std::size_t i = 0; i < variants.size(); ++i) {
        const Variant& v = variants[i];
        std::cout << v.key << "," << format_number(v.best, 6) << ","
                  << format_number(v.mean, 6) << "," << static_cast<long long>(v.ticks) << ","
                  << format_number(gflop / v.best, 4) << ","
                  << format_number(baseline / v.best, 4) << ","
                  << (checks[i].bit_exact ? 1 : 0) << "\n";
      }
      std::cout << "tile," << tile << "\nllc," << caches.llc << "\nline," << caches.line_size
                << "\nchecksum," << format_number(lab2::checksum(*reference->result), 10) << "\n";
      return all_match ? 0 : 1;
    }

    std::cout << "\n";
    {
      std::vector<std::vector<std::string>> rows;
      for (const Variant& v : variants) {
        rows.push_back({v.title, format_number(v.best, 3), format_number(v.mean, 3),
                        format_ticks(v.ticks), format_number(gflop / v.best, 2),
                        format_number(baseline / v.best, 2) + "×"});
      }
      print_table({38, 11, 11, 13, 10, 11},
                  {"Версия", "Лучшее, с", "Среднее, с", "Такты", "GFLOP/s", "Ускорение"}, rows);
      std::cout << "  Ускорение считается относительно C1 — версии только с векторизацией.\n";
    }

    std::cout << "\nОбоснование: проходы по матрицам и объём чтений\n";
    {
      const double size_a = static_cast<double>(m) * k_size * sizeof(lab2::Elem);
      const double size_b = static_cast<double>(k_size) * n * sizeof(lab2::Elem);
      const double size_c = static_cast<double>(m) * n * sizeof(lab2::Elem);
      std::vector<std::vector<std::string>> rows;
      for (const Variant& v : variants) {
        const double traffic = v.passes_a * size_a + v.passes_b * size_b + v.passes_c * size_c;
        rows.push_back({v.title, format_number(v.passes_a, 0), format_number(v.passes_b, 0),
                        format_number(v.passes_c, 0), format_bytes(traffic),
                        format_number(traffic / v.best / 1e9, 1)});
      }
      print_table({38, 10, 10, 10, 12, 13},
                  {"Версия", "проходов A", "проходов B", "обращ. C", "объём", "ГБ/с"}, rows);
      std::cout
          << "  Проход = сколько раз алгоритм целиком перечитывает матрицу. Проход реально\n"
             "  идёт в память только если переиспользуемый блок не помещается в кэш —\n"
             "  именно поэтому сторона тайла T считается из размера кэша.\n";
    }

    std::cout << "\nПроверка совпадения результатов (эталон — C1)\n";
    for (std::size_t i = 0; i < variants.size(); ++i) {
      if (variants[i].key == "c1") continue;
      print_field(variants[i].key + " vs C1",
                  checks[i].bit_exact
                      ? "СОВПАДАЮТ ПОБИТОВО"
                      : "РАСХОЖДЕНИЕ: " + std::to_string(checks[i].diff_count) +
                            " элементов, max |Δ| = " +
                            format_number(checks[i].max_abs_diff, 17),
                  20);
    }
    print_field("Контрольная сумма C", format_number(lab2::checksum(*reference->result), 10), 20);

    std::cout << "\nИтог\n";
    std::cout << (all_match
                      ? "  ✓ Результаты всех версий совпадают с C1 полностью (побитово).\n"
                      : "  ✗ Результаты РАЗЛИЧАЮТСЯ — требование задания не выполнено.\n");
    {
      const Variant* best = nullptr;
      for (const Variant& v : variants) {
        if (v.key == "naive") continue;
        if (best == nullptr || v.best < best->best) best = &v;
      }
      const Variant* plain = nullptr;
      for (const Variant& v : variants) {
        if (v.key == "naive") plain = &v;
      }
      if (plain != nullptr && best != nullptr) {
        std::cout << "  ✓ Относительно скалярной опорной версии C0 ускорение "
                  << format_number(plain->best / best->best, 1) << "×.\n";
      }
      if (best != nullptr && best->key != "c1") {
        std::cout << "  ✓ Оптимизация под кэш даёт " << format_number(baseline / best->best, 2)
                  << "× относительно версии только с векторизацией (быстрее всех — "
                  << best->key << ").\n";
      } else {
        std::cout << "  ! Блочные версии не обогнали C1 — увеличьте --size, матрицы ещё "
                     "помещаются в кэш.\n";
      }
      for (const Variant& v : variants) {
        if (v.key == "c2" && v.best < baseline) {
          std::cout << "  ✓ C2 (версия по заданию) быстрее C1 в "
                    << format_number(baseline / v.best, 2) << "× при тайле T=" << tile << ".\n";
        }
      }
    }
    std::cout << "\n";

    return all_match ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Ошибка: не удалось выделить память под матрицы такого размера ("
              << error.what() << ").\nПопробуйте меньший --size.\n";
    return 3;
  }
}
