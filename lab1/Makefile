CXX      ?= c++
BUILD    := build
TARGET   := $(BUILD)/lab1

BLK      ?= 8
SIZE     ?= 448
REPEAT   ?= 1
ARGS     ?=
EXTRA    ?=
ROW_STEP ?= 2

STD           := -std=c++17
RELEASE_FLAGS := -O3 -DNDEBUG
FP_FLAGS      := -ffp-contract=off
ARCH_FLAGS    := -mcpu=native
WARN_FLAGS    := -Wall -Wextra

IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -ci clang)
ifeq ($(IS_CLANG),0)
  NOVEC_FLAGS      := -fno-tree-vectorize -fno-tree-slp-vectorize
  VEC_REPORT_FLAGS := -fopt-info-vec -fopt-info-vec-missed
else
  NOVEC_FLAGS      := -fno-vectorize -fno-slp-vectorize
  VEC_REPORT_FLAGS := -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
                      -Rpass-analysis=loop-vectorize
endif

INFO_FLAGS := -DLAB1_CXXFLAGS='"$(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS)"' \
              -DLAB1_NOVEC_FLAGS='"$(NOVEC_FLAGS)"'

CXXFLAGS := $(STD) $(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS) $(WARN_FLAGS) \
            -Iinclude -DBLK=$(BLK) -DLAB1_ROW_STEP=$(ROW_STEP) $(EXTRA)

OBJS := $(BUILD)/main.o $(BUILD)/block_matrix.o $(BUILD)/multiply_simd.o \
        $(BUILD)/multiply_auto.o $(BUILD)/multiply_novec.o

HEADERS := include/block_matrix.hpp include/multiply.hpp include/timer.hpp

.PHONY: all run bench check asm vecreport no-pragma-check help clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(EXTRA) $(OBJS) -o $@
	@echo "Собрано: $@   (BLK=$(BLK), $(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS))"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/main.o: src/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INFO_FLAGS) -c $< -o $@

$(BUILD)/block_matrix.o: src/block_matrix.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/multiply_simd.o: src/multiply_simd.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/multiply_auto.o: src/multiply_scalar.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -DMUL_FN=multiply_auto -c $< -o $@

$(BUILD)/multiply_novec.o: src/multiply_scalar.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(NOVEC_FLAGS) -DMUL_FN=multiply_novec -c $< -o $@

run: $(TARGET)
	./$(TARGET) --size $(SIZE) --repeat $(REPEAT) $(ARGS)

bench: $(TARGET)
	./$(TARGET) --size $(SIZE) --repeat 3 $(ARGS)

check: $(TARGET)
	./$(TARGET) --size 48 --repeat 1

asm: | $(BUILD)
	@$(CXX) $(CXXFLAGS) -DMUL_FN=multiply_auto -S src/multiply_scalar.cpp -o $(BUILD)/multiply_auto.s
	@$(CXX) $(CXXFLAGS) $(NOVEC_FLAGS) -DMUL_FN=multiply_novec -S src/multiply_scalar.cpp -o $(BUILD)/multiply_novec.s
	@$(CXX) $(CXXFLAGS) -S src/multiply_simd.cpp -o $(BUILD)/multiply_simd.s
	@echo ""
	@echo "Ассемблерный листинг: $(BUILD)/*.s"
	@echo "Подсчёт инструкций умножения/сложения double:"
	@echo ""
	@echo "  Объектный файл                        векторных    скалярных"
	@echo "  ────────────────────────────────── ──────────── ────────────"
	@for f in multiply_novec multiply_auto multiply_simd; do \
	  v=$$(grep -cE '^[[:space:]]*(fmul|fadd|fmla)(\.2d[[:space:]]+v|[[:space:]]+v[0-9]+\.2d)' $(BUILD)/$$f.s || true); \
	  s=$$(grep -cE '^[[:space:]]*(fmul|fadd|fmadd)[[:space:]]+d[0-9]+' $(BUILD)/$$f.s || true); \
	  printf "  %-34s %12s %12s\n" "$$f.s" "$$v" "$$s"; \
	done
	@echo ""
	@echo "  векторных = fmul/fadd v<N>.2d (две операции double за инструкцию)"
	@echo "  скалярных = fmul/fadd d<N>    (одна операция double за инструкцию)"
	@echo ""

vecreport: | $(BUILD)
	@echo ""
	@echo "══ multiply_scalar.cpp, автовекторизация ВКЛЮЧЕНА ═════════════════════════"
	@$(CXX) $(CXXFLAGS) $(VEC_REPORT_FLAGS) -DMUL_FN=multiply_auto \
	    -c src/multiply_scalar.cpp -o $(BUILD)/vecreport_auto.o 2>&1 | grep -E 'remark|vector' || echo "  (замечаний нет)"
	@echo ""
	@echo "══ multiply_scalar.cpp, автовекторизация ВЫКЛЮЧЕНА ($(NOVEC_FLAGS)) ══"
	@$(CXX) $(CXXFLAGS) $(NOVEC_FLAGS) $(VEC_REPORT_FLAGS) -DMUL_FN=multiply_novec \
	    -c src/multiply_scalar.cpp -o $(BUILD)/vecreport_novec.o 2>&1 | grep -E 'remark|vector' || echo "  (замечаний нет)"
	@echo ""
	@echo "Внутренний цикл по col имеет константное число итераций ($(BLK)), поэтому"
	@echo "компилятор сначала разворачивает его целиком, и векторизацию выполняет"
	@echo "SLP-векторизатор, который отдельного замечания \"vectorized\" не печатает."
	@echo "Основное доказательство — дизассемблер: make asm"
	@echo ""

no-pragma-check:
	@if grep -rniE '#[[:space:]]*pragma.*(vector|simd|omp|ivdep)' src include; then \
	  echo "НАЙДЕНЫ прагмы векторизации — это запрещено условием."; exit 1; \
	else \
	  echo "OK: прагм, управляющих векторизацией, в исходниках нет."; \
	  echo "    Векторизация отключается только ключами: $(NOVEC_FLAGS)"; \
	fi

help:
	@echo ""
	@echo "Лабораторная работа №1 — умножение блочных матриц (double, блок $(BLK)×$(BLK))"
	@echo ""
	@echo "  make                 собрать (Release: $(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS))"
	@echo "  make run             собрать и запустить (SIZE=$(SIZE), REPEAT=$(REPEAT))"
	@echo "  make bench           прогон в три повтора"
	@echo "  make check           быстрая проверка совпадения C1 и C2"
	@echo "  make asm             ассемблерный листинг + подсчёт векторных инструкций"
	@echo "  make vecreport       отчёт компилятора о векторизации циклов"
	@echo "  make no-pragma-check убедиться, что прагм векторизации в коде нет"
	@echo "  make clean           удалить build/"
	@echo ""
	@echo "Переменные: SIZE=<L или L,M,N>  REPEAT=<N>  BLK=<N>  ROW_STEP=<2|4>  ARGS='<флаги>'  EXTRA='<флаги>'"
	@echo "Например:   make run SIZE=320 REPEAT=3"
	@echo "            make run ARGS='--skip-novec'"
	@echo "            ./$(TARGET) --help"
	@echo ""

clean:
	rm -rf $(BUILD)
