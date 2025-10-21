cmake_minimum_required(VERSION 3.15)
project(tensr VERSION 0.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TENSR_BUILD_CUDA "Build with CUDA support" OFF)
option(TENSR_BUILD_TESTS "Build tests" ON)
option(TENSR_BUILD_EXAMPLES "Build examples" ON)

# Check if building from source or using prebuilt library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
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
else()
    # Prebuilt library package - only provide headers for user integration
    message(STATUS "Tensr prebuilt package detected. Include headers are available.")
    message(STATUS "Link against lib/tensr.lib (Windows) or lib/libtensr.a (Unix) in your project.")
    add_library(tensr INTERFACE)
    target_include_directories(tensr INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# Tests (only when building from source)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(TENSR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.c")
        add_executable(tests tests/test_tensor.c)
        target_link_libraries(tests PRIVATE tensr)
        target_include_directories(tests PRIVATE include)
    endif()

    # Examples (only when building from source)
    if(TENSR_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
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
endif()

# Installation (only when building from source)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    install(TARGETS tensr
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        install(DIRECTORY include/
            DESTINATION include
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        )
    endif()
    
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
endif()
