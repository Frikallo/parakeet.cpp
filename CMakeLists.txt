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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ── Options ──────────────────────────────────────────────────────────────────
option(PARAKEET_BUILD_CLI "Build the parakeet CLI binary" ON)
option(PARAKEET_BUILD_TESTS "Build parakeet tests" ON)
option(PARAKEET_BUILD_BENCHMARKS "Build parakeet benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build shared library instead of static" OFF)

# ── Axiom ────────────────────────────────────────────────────────────────────
set(AXIOM_BUILD_TESTS OFF)
set(AXIOM_BUILD_EXAMPLES OFF)
if(PARAKEET_BUILD_BENCHMARKS)
    set(AXIOM_BUILD_BENCHMARKS ON)
else()
    set(AXIOM_BUILD_BENCHMARKS OFF)
endif()

add_subdirectory(third_party/axiom)

# ── Parakeet library ─────────────────────────────────────────────────────────
add_library(parakeet_lib
    src/encoder.cpp
    src/lstm.cpp
    src/ctc.cpp
    src/rnnt.cpp
    src/tdt.cpp
    src/tdt_ctc.cpp
    src/audio.cpp
    src/audio_io.cpp
    src/vocab.cpp
    src/timestamp.cpp
    src/streaming_encoder.cpp
    src/eou.cpp
    src/nemotron.cpp
    src/transformer.cpp
    src/sortformer.cpp
    src/diarize.cpp
    src/phrase_boost.cpp
)

add_library(Parakeet::parakeet ALIAS parakeet_lib)

target_link_libraries(parakeet_lib PUBLIC Axiom::axiom)

target_include_directories(parakeet_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dr_libs
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

target_compile_options(parakeet_lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

set_target_properties(parakeet_lib PROPERTIES
    EXPORT_NAME parakeet
    OUTPUT_NAME parakeet
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ── CLI binary ───────────────────────────────────────────────────────────────
if(PARAKEET_BUILD_CLI)
    add_executable(parakeet src/main.cpp)
    target_link_libraries(parakeet PRIVATE parakeet_lib)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
install(TARGETS parakeet_lib
    EXPORT ParakeetTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(PARAKEET_BUILD_CLI)
    install(TARGETS parakeet RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ParakeetTargets
    FILE ParakeetTargets.cmake
    NAMESPACE Parakeet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Parakeet
)

configure_package_config_file(
    cmake/ParakeetConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ParakeetConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Parakeet
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ParakeetConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ParakeetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ParakeetConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Parakeet
)

# ── pkg-config (Unix) ───────────────────────────────────────────────────────
if(UNIX)
    configure_file(
        cmake/parakeet.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/parakeet.pc"
        @ONLY
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/parakeet.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
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
    add_test(NAME parakeet_tests COMMAND parakeet_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# ── Benchmarks ───────────────────────────────────────────────────────────────
if(PARAKEET_BUILD_BENCHMARKS)
    add_executable(parakeet_bench src/bench.cpp)
    target_link_libraries(parakeet_bench PRIVATE parakeet_lib benchmark::benchmark)
endif()

# ── Configuration summary ───────────────────────────────────────────────────
message(STATUS "")
message(STATUS "parakeet ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CLI:              ${PARAKEET_BUILD_CLI}")
message(STATUS "  Tests:            ${PARAKEET_BUILD_TESTS}")
message(STATUS "  Benchmarks:       ${PARAKEET_BUILD_BENCHMARKS}")
message(STATUS "  Shared libs:      ${BUILD_SHARED_LIBS}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
