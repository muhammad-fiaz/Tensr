cmake_minimum_required(VERSION 3.15)
project(tensr VERSION 0.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TENSR_BUILD_CUDA "Build with CUDA support" OFF)
option(TENSR_BUILD_TESTS "Build tests" ON)
option(TENSR_BUILD_EXAMPLES "Build examples" ON)

# Source files
set(TENSR_SOURCES
    src/core/tensor.c
    src/core/array.c
    src/core/tensor.cpp
    src/ops/arithmetic.c
    src/ops/reduction.c
    src/linalg/linalg.c
    src/random/random.c
    src/io/io.c
    src/fft/fft.c
    src/backend/device.c
)

# CUDA support
if(TENSR_BUILD_CUDA)
    enable_language(CUDA)
    list(APPEND TENSR_SOURCES src/backend/cuda/cuda_ops.cu)
endif()

# Main library
add_library(tensr STATIC ${TENSR_SOURCES})
target_include_directories(tensr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
if(TENSR_BUILD_TESTS)
    add_executable(tests tests/test_tensor.c)
    target_link_libraries(tests PRIVATE tensr)
    target_include_directories(tests PRIVATE include)
endif()

# Examples
if(TENSR_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_usage.c")
        add_executable(example_c examples/basic_usage.c)
        target_link_libraries(example_c PRIVATE tensr)
        target_include_directories(example_c PRIVATE include)
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp_usage.cpp")
        add_executable(example_cpp examples/cpp_usage.cpp)
        target_link_libraries(example_cpp PRIVATE tensr)
        target_include_directories(example_cpp PRIVATE include)
    endif()
endif()

# Installation
install(TARGETS tensr
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tensrConfigVersion.cmake"
    DESTINATION lib/cmake/tensr
)
