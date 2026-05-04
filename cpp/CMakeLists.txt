cmake_minimum_required(VERSION 3.18)
project(nano_kws_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------
# Locate ONNX Runtime.
#
# Two ways to point CMake at it:
#
#   1. Set ONNXRUNTIME_ROOT (recommended): a directory that contains
#      `include/onnxruntime_cxx_api.h` and `lib/onnxruntime{.so,.dylib,.dll}`.
#      e.g. extract a release zip from https://github.com/microsoft/onnxruntime
#      and pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.18.0
#
#   2. Drop a system-installed onnxruntime in a place CMake can find via
#      find_package(onnxruntime). Most package-manager builds work.
# ----------------------------------------------------------------------------

if(DEFINED ONNXRUNTIME_ROOT)
    message(STATUS "Using ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            IMPORTED_IMPLIB   "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        )
    elseif(APPLE)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib"
        )
    else()
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        )
    endif()
else()
    find_package(onnxruntime REQUIRED)
endif()

add_executable(nano_kws_infer infer.cpp)
target_link_libraries(nano_kws_infer PRIVATE onnxruntime)

if(MSVC)
    target_compile_options(nano_kws_infer PRIVATE /W4 /permissive-)
else()
    target_compile_options(nano_kws_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()
