NVCC     ?= nvcc
ARCH     ?= native
FMAD     ?= false
BUILD    := build
TARGET   := $(BUILD)/lab3

ARGS     ?=
EXTRA    ?=

NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -fmad=$(FMAD) \
            -Xcompiler -O3,-ffp-contract=off,-Wall,-Wextra -Iinclude $(EXTRA)

SOURCES  := $(wildcard src/*.cu) $(wildcard src/*.cpp)
OBJS     := $(patsubst src/%,$(BUILD)/%.o,$(SOURCES))
HEADERS  := $(wildcard include/*)

.PHONY: all run quick image ptx help clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -arch=$(ARCH) $(OBJS) -o $@
	@echo "Собрано: $@   (-O3 -arch=$(ARCH) -fmad=$(FMAD))"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/% $(HEADERS) | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

quick: $(TARGET)
	./$(TARGET) --size 1024 1024 --iter 300 $(ARGS)

image: $(TARGET)
	./$(TARGET) --out julia.ppm $(ARGS)

ptx: | $(BUILD)
	$(NVCC) $(NVFLAGS) -ptx src/kernels.cu -o $(BUILD)/kernels.ptx
	@grep -c "fma.rn.f32" $(BUILD)/kernels.ptx | sed 's/^/fma.rn.f32 в PTX: /' || true

help:
	@echo "make          сборка (нужен nvcc; ARCH=sm_86 и т.п., если native не поддерживается)"
	@echo "make run      основной прогон: 4096×2304, Nmax=1000, CPU-эталон + GPU"
	@echo "make quick    быстрый прогон 1024×1024"
	@echo "make image    прогон с сохранением julia.ppm"
	@echo "make ptx      PTX ядер и проверка, что FMA не появилось"
	@echo "ARGS='--c 0.285 0.01 --no-cpu'  передать ключи программе"

clean:
	rm -rf $(BUILD) julia.ppm
