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

# turn off axiom tests, examples
set(AXIOM_BUILD_TESTS OFF)
set(AXIOM_BUILD_EXAMPLES OFF)

# Enable axiom benchmarks only when building parakeet benchmarks
option(PARAKEET_BUILD_BENCHMARKS "Build parakeet benchmarks" ON)
if(PARAKEET_BUILD_BENCHMARKS)
    set(AXIOM_BUILD_BENCHMARKS ON)
else()
    set(AXIOM_BUILD_BENCHMARKS OFF)
endif()

add_subdirectory(third_party/axiom)

# Parakeet library
add_library(parakeet_lib STATIC
    src/encoder.cpp
    src/lstm.cpp
    src/ctc.cpp
    src/rnnt.cpp
    src/tdt.cpp
    src/tdt_ctc.cpp
    src/wav.cpp
    src/audio.cpp
    src/audio_io.cpp
    src/vocab.cpp
    src/timestamp.cpp
    src/streaming_encoder.cpp
    src/eou.cpp
    src/nemotron.cpp
    src/transformer.cpp
    src/sortformer.cpp
)
target_link_libraries(parakeet_lib PUBLIC Axiom::axiom)
target_include_directories(parakeet_lib PUBLIC include)
target_include_directories(parakeet_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/dr_libs
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# Demo binary
add_executable(parakeet src/main.cpp)
target_link_libraries(parakeet PRIVATE parakeet_lib)

install(TARGETS parakeet parakeet_lib DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Tests
option(PARAKEET_BUILD_TESTS "Build parakeet tests" ON)
if(PARAKEET_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(parakeet_tests tests/test_all.cpp)
    target_link_libraries(parakeet_tests PRIVATE parakeet_lib GTest::gtest GTest::gtest_main)
    add_test(NAME parakeet_tests COMMAND parakeet_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Benchmarks
if(PARAKEET_BUILD_BENCHMARKS)
    add_executable(parakeet_bench src/bench.cpp)
    target_link_libraries(parakeet_bench PRIVATE parakeet_lib benchmark::benchmark)
endif()
