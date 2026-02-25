cmake_minimum_required(VERSION 3.20)

project(Parakeet
    VERSION 0.1.0
    DESCRIPTION "Portable Parakeet Inference Engine"
    LANGUAGES CXX
)

# C++20 is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# turn off axiom tests, examples, benchmarks
set(AXIOM_BUILD_TESTS OFF)
set(AXIOM_BUILD_EXAMPLES OFF)
set(AXIOM_BUILD_BENCHMARKS OFF)
add_subdirectory(third_party/axiom)

# Parakeet library
add_library(parakeet_lib STATIC
    src/encoder.cpp
    src/lstm.cpp
    src/ctc.cpp
    src/rnnt.cpp
    src/tdt.cpp
    src/tdt_ctc.cpp
)
target_link_libraries(parakeet_lib PUBLIC Axiom::axiom)
target_include_directories(parakeet_lib PUBLIC include)

# Demo binary
add_executable(parakeet src/main.cpp)
target_link_libraries(parakeet PRIVATE parakeet_lib)

install(TARGETS parakeet parakeet_lib DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
