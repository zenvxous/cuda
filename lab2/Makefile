CXX      ?= c++
BUILD    := build
TARGET   := $(BUILD)/lab2

SIZE     ?= 2560
REPEAT   ?= 1
ARGS     ?=
EXTRA    ?=

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

INFO_FLAGS := -DLAB2_CXXFLAGS='"$(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS)"'

CXXFLAGS := $(STD) $(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS) $(WARN_FLAGS) -Iinclude $(EXTRA)

SOURCES := $(wildcard src/*.cpp)
OBJS    := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SOURCES))
HEADERS := $(wildcard include/*.hpp)

.PHONY: all run bench check probe caches asm vecreport no-pragma-check help clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(EXTRA) $(OBJS) -o $@
	@echo "Собрано: $@   ($(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS))"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/main.o: src/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INFO_FLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) --size $(SIZE) --repeat $(REPEAT) $(ARGS)

bench: $(TARGET)
	./$(TARGET) --size $(SIZE) --repeat 3 $(ARGS)

check: $(TARGET)
	./$(TARGET) --size 320

probe: $(TARGET)
	./$(TARGET) --probe

caches: $(TARGET)
	@./$(TARGET) --size $(SIZE) --info

x86sim: $(TARGET)
	./$(TARGET) --size $(SIZE) --llc 8M --l2 1M --l1 48K --line 64 $(ARGS)

asm: | $(BUILD)
	@$(CXX) $(CXXFLAGS) -S src/mul_vec.cpp -o $(BUILD)/mul_vec.s
	@$(CXX) $(CXXFLAGS) -S src/mul_block.cpp -o $(BUILD)/mul_block.s
	@$(CXX) $(CXXFLAGS) -S src/mul_block_multi.cpp -o $(BUILD)/mul_block_multi.s
	@$(CXX) $(CXXFLAGS) -S src/mul_naive.cpp -o $(BUILD)/mul_naive.s
	@echo ""
	@echo "Ассемблерный листинг: $(BUILD)/*.s"
	@echo ""
	@echo "  Объектный файл                        векторных    скалярных"
	@echo "  ────────────────────────────────── ──────────── ────────────"
	@for f in mul_naive mul_vec mul_block mul_block_multi; do \
	  v=$$(grep -cE '^[[:space:]]*(fmul|fadd|fmla)(\.2d[[:space:]]+v|[[:space:]]+v[0-9]+\.2d)' $(BUILD)/$$f.s || true); \
	  s=$$(grep -cE '^[[:space:]]*(fmul|fadd|fmadd)[[:space:]]+d[0-9]+' $(BUILD)/$$f.s || true); \
	  printf "  %-34s %12s %12s\n" "$$f.s" "$$v" "$$s"; \
	done
	@echo ""
	@echo "  векторных = fmul/fadd v<N>.2d (две операции double за инструкцию)"
	@echo "  скалярных = fmul/fadd d<N>    (одна операция double за инструкцию)"
	@echo ""

vecreport: | $(BUILD)
	@$(CXX) $(CXXFLAGS) $(VEC_REPORT_FLAGS) -c src/mul_vec.cpp -o $(BUILD)/vecreport.o 2>&1 | grep -E 'remark' || echo "  (замечаний нет)"

no-pragma-check:
	@if grep -rniE '#[[:space:]]*pragma.*(vector|simd|omp|ivdep)' src include; then \
	  echo "НАЙДЕНЫ прагмы векторизации."; exit 1; \
	else \
	  echo "OK: прагм, управляющих векторизацией, в исходниках нет."; \
	fi

help:
	@echo ""
	@echo "Лабораторная работа №2 — оптимизация умножения матриц под кэш"
	@echo ""
	@echo "  make                 собрать (Release: $(RELEASE_FLAGS) $(FP_FLAGS) $(ARCH_FLAGS))"
	@echo "  make run             прогон (SIZE=$(SIZE), REPEAT=$(REPEAT))"
	@echo "  make bench           прогон в три повтора"
	@echo "  make check           малый размер, быстрая проверка совпадения"
	@echo "  make probe           профиль иерархии кэша"
	@echo "  make caches          показать обнаруженные кэши и размеры блоков"
	@echo "  make x86sim          прогон с кэшами типовой x86-машины (три уровня блокировки)"
	@echo "  make asm             подсчёт векторных инструкций в ядрах"
	@echo "  make vecreport       отчёт компилятора о векторизации"
	@echo "  make no-pragma-check прагм векторизации в коде нет"
	@echo "  make clean           удалить build/"
	@echo ""
	@echo "Переменные: SIZE=<N|M,K,N>  REPEAT=<N>  ARGS='<флаги>'  EXTRA='<флаги компилятора>'"
	@echo "Например:   make run SIZE=3072 REPEAT=3"
	@echo "            ./$(TARGET) --help"
	@echo ""

clean:
	rm -rf $(BUILD)
