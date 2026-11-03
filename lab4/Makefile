NVCC     ?= nvcc
ARCH     ?= native
FMAD     ?= false
BUILD    := build
TARGET   := $(BUILD)/lab4

ARGS     ?=
EXTRA    ?=

NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -fmad=$(FMAD) \
            -Xcompiler -O3,-ffp-contract=off,-Wall,-Wextra -Iinclude $(EXTRA)

SOURCES  := $(wildcard src/*.cu)
OBJS     := $(patsubst src/%,$(BUILD)/%.o,$(SOURCES))
HEADERS  := $(wildcard include/*)

.PHONY: all run quick ptx help clean

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
	./$(TARGET) --n 100000 $(ARGS)

ptx: | $(BUILD)
	$(NVCC) $(NVFLAGS) -ptx src/kernels.cu -o $(BUILD)/kernels.ptx
	@grep -c "fma.rn.f32" $(BUILD)/kernels.ptx | sed 's/^/fma.rn.f32 в PTX: /' || true

help:
	@echo "make          сборка (нужен nvcc; ARCH=sm_86 и т.п., если native не поддерживается)"
	@echo "make run      основной прогон: N = 10^6, CPU + G1, G2, G3 (4 блока), CUB"
	@echo "make quick    прогон с N = 10^5"
	@echo "make ptx      PTX ядер и проверка, что FMA не появилось"
	@echo "ARGS='--n 1000 --block 128'  передать ключи программе"

clean:
	rm -rf $(BUILD)
