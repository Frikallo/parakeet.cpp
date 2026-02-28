# Contributing to parakeet.cpp

## Build

Requires C++20, CMake 3.20+, and macOS 13+ (for Metal GPU acceleration).

```bash
git clone --recursive https://github.com/frikallo/parakeet.cpp
cd parakeet.cpp
make build
```

To build without the CLI binary:

```bash
make build CLI=OFF
```

## Test

```bash
make test
```

## Code Style

The project uses `clang-format`. Format your code before committing:

```bash
make format
```

Check formatting without modifying files:

```bash
make format-check
```

CI enforces formatting on all pull requests.

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused and atomic.
3. Run `make format` and `make test` before pushing.
4. Open a pull request against `main` with a clear description of the change.

## Project Structure

```
include/parakeet/   Public headers (installed)
src/                Implementation files + CLI entry point (main.cpp)
tests/              Google Test test files
scripts/            Weight conversion and utility scripts
third_party/        Submodules (axiom, dr_libs, stb)
cmake/              CMake packaging files
```
