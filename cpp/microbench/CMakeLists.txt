cmake_minimum_required(VERSION 3.18)
project(conv_kernels C)

# This builds a single shared library (conv_kernels.dll / .so / .dylib)
# that the Python orchestrator in scripts/conv_microbench.py loads via
# ctypes. No external dependencies — it's just plain C with AVX2
# intrinsics.

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(conv_kernels SHARED conv_kernels.c)
target_include_directories(conv_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Strip the lib prefix so we get "conv_kernels.dll" / "conv_kernels.so"
# / "conv_kernels.dylib" everywhere — the Python loader assumes that
# exact name.
set_target_properties(conv_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "conv_kernels"
)

# Compiler flags.
#
# The whole point of this microbench is to give the AVX2 path a fair
# shake, so we crank the optimiser, enable AVX2 + FMA, and turn on
# fast-math (no NaN / inf semantics, no signed-zero preservation —
# the only caller is a benchmark that doesn't depend on those).
if(MSVC)
    target_compile_options(conv_kernels PRIVATE /O2 /arch:AVX2 /fp:fast /W4)
else()
    target_compile_options(conv_kernels PRIVATE -O3 -mavx2 -mfma -ffast-math -Wall -Wextra -Wpedantic)
endif()
