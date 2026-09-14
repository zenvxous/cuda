#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "block_matrix.hpp"
#include "multiply.hpp"
#include "timer.hpp"

#ifndef LAB1_CXXFLAGS
#define LAB1_CXXFLAGS "(не заданы)"
#endif
#ifndef LAB1_NOVEC_FLAGS
#define LAB1_NOVEC_FLAGS "(не заданы)"
#endif

namespace {

using lab1::BlockMatrix;
using lab1::kBlk;

struct Options {
  int rows = 448;
  int inner = 448;
  int cols = 448;
  int repeat = 1;
  bool skip_novec = false;
  bool csv = false;
  std::uint64_t seed = 20240501;
};

struct Variant {
  std::string title;
  std::string csv_key;
  double best_seconds = 0.0;
  double mean_seconds = 0.0;
  double best_ticks = 0.0;
  bool executed = false;
};

void print_usage(const char* program) {
  std::cout <<
      "Лабораторная работа №1 — умножение матриц, элементы которых суть матрицы "
            << kBlk << "×" << kBlk << " (double).\n\n"
      "Использование:\n"
      "  " << program << " [параметры]\n\n"
      "Параметры:\n"
      "  -s, --size L[,M,N]   размеры ВНЕШНИХ матриц: A[L×M] × B[M×N] = C[L×N].\n"
      "                       Одно число задаёт квадратный случай. По умолчанию 448.\n"
      "  -r, --repeat K       прогонов каждой версии (в отчёт идут лучший и средний).\n"
      "                       По умолчанию 1.\n"
      "      --skip-novec     не запускать невекторизованную версию (она самая долгая).\n"
      "      --seed N         зерно генератора исходных данных. По умолчанию 20240501.\n"
      "      --csv            машиночитаемый вывод для отчёта.\n"
      "  -h, --help           эта справка.\n\n"
      "Примеры:\n"
      "  " << program << "                      прогон с настройками по умолчанию\n"
      "  " << program << " --size 64 --repeat 3 быстрая проверка, три прогона\n"
      "  " << program << " --size 200,256,300   неквадратные внешние матрицы\n";
}

bool parse_positive(const char* text, long long& out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) return false;
  out = value;
  return true;
}

bool parse_size(const char* text, Options& options) {
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

bool parse_options(int argc, char** argv, Options& options, int& exit_code) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_next = (i + 1 < argc);
    long long value = 0;

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      exit_code = 0;
      return false;
    }
    if (arg == "--skip-novec") { options.skip_novec = true; continue; }
    if (arg == "--csv") { options.csv = true; continue; }

    if (arg == "-s" || arg == "--size") {
      if (has_next && parse_size(argv[++i], options)) continue;
      std::cerr << "Ошибка: --size ожидает L или L,M,N (положительные числа).\n";
    } else if (arg == "-r" || arg == "--repeat") {
      if (has_next && parse_positive(argv[++i], value) && value <= 1000) {
        options.repeat = static_cast<int>(value);
        continue;
      }
      std::cerr << "Ошибка: --repeat ожидает число от 1 до 1000.\n";
    } else if (arg == "--seed") {
      if (has_next && parse_positive(argv[++i], value)) {
        options.seed = static_cast<std::uint64_t>(value);
        continue;
      }
      std::cerr << "Ошибка: --seed ожидает положительное число.\n";
    } else {
      std::cerr << "Ошибка: неизвестный параметр \"" << arg << "\".\n"
                << "Подсказка: " << argv[0] << " --help\n";
    }
    exit_code = 2;
    return false;
  }
  return true;
}

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

void print_field(const std::string& name, const std::string& value, std::size_t width = 22) {
  std::string dotted = name + " ";
  while (visual_width(dotted) < width) dotted += ".";
  std::cout << "  " << dotted << " " << value << "\n";
}

template <typename Fn>
void run_variant(Variant& variant, int repeat, bool quiet, Fn&& multiply) {
  double best = 0.0;
  double total = 0.0;
  double best_ticks = 0.0;

  for (int run = 0; run < repeat; ++run) {
    if (!quiet) {
      std::cout << "  " << pad_right(variant.title, 38) << " прогон " << (run + 1) << "/"
                << repeat << " … " << std::flush;
    }

    const lab1::Measurement measurement = lab1::measure(multiply);
    total += measurement.seconds;
    if (run == 0 || measurement.seconds < best) {
      best = measurement.seconds;
      best_ticks = static_cast<double>(measurement.ticks);
    }

    if (!quiet) std::cout << format_number(measurement.seconds, 3) << " с\n";
  }

  variant.best_seconds = best;
  variant.mean_seconds = total / repeat;
  variant.best_ticks = best_ticks;
  variant.executed = true;
}

void print_table(const std::vector<Variant>& variants, double gflop, double baseline) {
  const std::size_t widths[] = {38, 11, 11, 13, 10, 11};

  auto rule = [&](const char* left, const char* mid, const char* right) {
    std::cout << left;
    for (std::size_t i = 0; i < 6; ++i) {
      std::cout << repeat_utf8("─", widths[i] + 2) << (i + 1 == 6 ? right : mid);
    }
    std::cout << "\n";
  };
  auto row = [&](const std::string(&cells)[6]) {
    std::cout << "│ " << pad_right(cells[0], widths[0]);
    for (std::size_t i = 1; i < 6; ++i) std::cout << " │ " << pad_left(cells[i], widths[i]);
    std::cout << " │\n";
  };

  rule("┌", "┬", "┐");
  row({"Версия", "Лучшее, с", "Среднее, с", "Такты", "GFLOP/s", "Ускорение"});
  rule("├", "┼", "┤");
  for (const Variant& variant : variants) {
    if (!variant.executed) {
      row({variant.title, "—", "—", "—", "—", "пропущена"});
      continue;
    }
    row({variant.title, format_number(variant.best_seconds, 3),
         format_number(variant.mean_seconds, 3), format_ticks(variant.best_ticks),
         format_number(gflop / variant.best_seconds, 2),
         format_number(baseline / variant.best_seconds, 2) + "×"});
  }
  rule("└", "┴", "┘");
}

}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);

  Options options;
  int exit_code = 0;
  if (!parse_options(argc, argv, options, exit_code)) return exit_code;

  const bool quiet = options.csv;
  const double gflop = 2.0 * options.rows * options.inner * options.cols * kBlk * kBlk * kBlk / 1e9;

  try {
    BlockMatrix a(options.rows, options.inner);
    BlockMatrix b(options.inner, options.cols);
    BlockMatrix c_novec(options.rows, options.cols);
    BlockMatrix c_auto(options.rows, options.cols);
    BlockMatrix c_simd(options.rows, options.cols);

    if (!quiet) {
      std::cout << "\n╔" << repeat_utf8("═", 76) << "╗\n"
                << "║ " << pad_right("ЛАБОРАТОРНАЯ РАБОТА №1 — умножение блочных матриц", 75)
                << "║\n║ "
                << pad_right("Вариант: double, малая матрица " + std::to_string(kBlk) + "×" +
                                 std::to_string(kBlk), 75)
                << "║\n╚" << repeat_utf8("═", 76) << "╝\n\n";

      std::cout << "Сборка\n";
      print_field("Архитектура", "x86-64");
      print_field("Ручной SIMD", "SSE2 (регистры xmm, 128 бит = 2 × double)");
      print_field("Компилятор", __VERSION__);
      print_field("Флаги (общие)", LAB1_CXXFLAGS);
      print_field("Флаги (без вект.)", std::string("+ ") + LAB1_NOVEC_FLAGS);
      print_field("Счётчик тактов", "__rdtsc() (такты процессора)");
      print_field("Часы", "std::chrono::high_resolution_clock");

      std::cout << "\nЗадача\n";
      print_field("Внешние матрицы",
                  "A[" + std::to_string(options.rows) + "×" + std::to_string(options.inner) +
                      "] × B[" + std::to_string(options.inner) + "×" +
                      std::to_string(options.cols) + "] = C[" + std::to_string(options.rows) +
                      "×" + std::to_string(options.cols) + "]");
      print_field("Элемент матрицы",
                  "матрица " + std::to_string(kBlk) + "×" + std::to_string(kBlk) + " (double)");
      print_field("Полный размер",
                  std::to_string(static_cast<long long>(options.rows) * kBlk) + "×" +
                      std::to_string(static_cast<long long>(options.inner) * kBlk) + " × " +
                      std::to_string(static_cast<long long>(options.inner) * kBlk) + "×" +
                      std::to_string(static_cast<long long>(options.cols) * kBlk) +
                      " элементов double");
      print_field("Память",
                  format_bytes(static_cast<double>(a.bytes()) + static_cast<double>(b.bytes()) +
                               3.0 * static_cast<double>(c_auto.bytes())) +
                      " (A + B + 3 варианта C)");
      print_field("Объём вычислений", format_number(gflop, 2) + " GFLOP");
      print_field("Прогонов на версию", std::to_string(options.repeat));

      std::cout << "\nПодготовка исходных данных … " << std::flush;
    }

    lab1::fill_random(a, options.seed);
    lab1::fill_random(b, options.seed * 6364136223846793005ULL + 1442695040888963407ULL);
    lab1::fill_zero(c_novec);
    lab1::fill_zero(c_auto);
    lab1::fill_zero(c_simd);

    if (!quiet) std::cout << "готово\n\nЗамеры\n";

    Variant novec{"C1  скаляр, векторизация ВЫКЛ", "scalar_novec", 0, 0, 0, false};
    Variant automatic{"C1  скаляр, автовекторизация", "scalar_auto", 0, 0, 0, false};
    Variant simd{"C2  ручной SIMD (SSE2)", "manual_simd", 0, 0, 0, false};

    if (!options.skip_novec) {
      run_variant(novec, options.repeat, quiet, [&] { lab1::multiply_novec(a, b, c_novec); });
    }
    run_variant(automatic, options.repeat, quiet, [&] { lab1::multiply_auto(a, b, c_auto); });
    run_variant(simd, options.repeat, quiet, [&] { lab1::multiply_simd(a, b, c_simd); });

    const lab1::CompareResult auto_vs_simd = lab1::compare(c_auto, c_simd);
    const lab1::CompareResult auto_vs_novec =
        options.skip_novec ? lab1::CompareResult{true, 0.0, 0} : lab1::compare(c_auto, c_novec);
    const bool all_match = auto_vs_simd.bit_exact && auto_vs_novec.bit_exact;
    const double baseline = novec.executed ? novec.best_seconds : automatic.best_seconds;

    if (options.csv) {
      std::cout << "version,best_seconds,mean_seconds,ticks,gflops,speedup\n";
      for (const Variant* variant : {&novec, &automatic, &simd}) {
        if (!variant->executed) continue;
        std::cout << variant->csv_key << "," << format_number(variant->best_seconds, 6) << ","
                  << format_number(variant->mean_seconds, 6) << ","
                  << static_cast<long long>(variant->best_ticks) << ","
                  << format_number(gflop / variant->best_seconds, 4) << ","
                  << format_number(baseline / variant->best_seconds, 4) << "\n";
      }
      std::cout << "bit_exact_c1_vs_c2," << (auto_vs_simd.bit_exact ? 1 : 0) << "\n"
                << "bit_exact_c1vec_vs_c1novec," << (auto_vs_novec.bit_exact ? 1 : 0) << "\n"
                << "max_abs_diff," << auto_vs_simd.max_abs_diff << "\n"
                << "checksum," << format_number(lab1::checksum(c_auto), 10) << "\n";
      return all_match ? 0 : 1;
    }

    std::cout << "\n";
    print_table({novec, automatic, simd}, gflop, baseline);
    std::cout << "  Ускорение считается относительно "
              << (novec.executed ? "невекторизованной версии." : "автовекторизованной версии.")
              << "\n\nПроверка совпадения результатов\n";

    print_field("C1(вект.) vs C1(без вект.)",
                options.skip_novec ? "не проверялось (--skip-novec)"
                                   : (auto_vs_novec.bit_exact ? "СОВПАДАЮТ ПОБИТОВО"
                                                              : "РАСХОЖДЕНИЕ"),
                30);
    print_field("C1 vs C2 (ручной SIMD)",
                auto_vs_simd.bit_exact ? "СОВПАДАЮТ ПОБИТОВО" : "РАСХОЖДЕНИЕ", 30);
    print_field("Максимальное |C1 - C2|", format_number(auto_vs_simd.max_abs_diff, 1), 30);
    print_field("Различающихся элементов", std::to_string(auto_vs_simd.diff_count), 30);
    print_field("Контрольная сумма C", format_number(lab1::checksum(c_auto), 10), 30);

    std::cout << "\nИтог\n";
    std::cout << (all_match
                      ? "  ✓ Результаты всех версий совпадают полностью (побитово).\n"
                      : "  ✗ Результаты версий РАЗЛИЧАЮТСЯ — требование задания не выполнено.\n");

    if (simd.executed && automatic.executed) {
      const double ratio = automatic.best_seconds / simd.best_seconds;
      const std::string tail = " (" + format_number(ratio, 2) + "×).\n";
      if (ratio >= 1.0) std::cout << "  ✓ Ручной SIMD не медленнее автовекторизации" << tail;
      else if (ratio >= 0.98) std::cout << "  ✓ Ручной SIMD и автовекторизация равны в пределах погрешности" << tail;
      else std::cout << "  ✗ Ручной SIMD МЕДЛЕННЕЕ автовекторизации" << tail;
    }
    if (automatic.executed) {
      std::cout << (automatic.best_seconds < 3.0
                        ? "  ! C1 считается быстрее нескольких секунд — для отчёта увеличьте --size.\n"
                        : "  ✓ C1 считается дольше нескольких секунд, как требует задание.\n");
    }
    std::cout << "\n";

    return all_match ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Ошибка: не удалось выделить память под матрицы такого размера ("
              << error.what() << ").\nПопробуйте меньший --size.\n";
    return 3;
  }
}
