cmake_minimum_required(VERSION 3.20)

project(Parakeet
    VERSION 0.0.1
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

add_executable(parakeet src/parakeet.cpp)
target_link_libraries(parakeet PRIVATE Axiom::axiom)
target_include_directories(parakeet PRIVATE include)

install(TARGETS parakeet DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)