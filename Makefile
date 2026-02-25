BUILD_DIR   := build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release

# Use Ninja if available, otherwise Unix Makefiles
ifneq ($(shell which ninja 2>/dev/null),)
    GENERATOR    := Ninja
    BUILD_CMD    := ninja -C $(BUILD_DIR)
else
    GENERATOR    := "Unix Makefiles"
    NPROC        := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    BUILD_CMD    := $(MAKE) -C $(BUILD_DIR) -j$(NPROC)
endif

.PHONY: all configure build run debug format format-check clean

all: build

configure:
	cmake -B $(BUILD_DIR) -G $(GENERATOR) $(CMAKE_FLAGS)

build: configure
	$(BUILD_CMD)

debug: CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Debug
debug: configure
	$(BUILD_CMD)

run: build
	./$(BUILD_DIR)/parakeet

format:
	find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

format-check:
	find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run --Werror

clean:
	rm -rf $(BUILD_DIR)
