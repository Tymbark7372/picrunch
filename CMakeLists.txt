cmake_minimum_required(VERSION 3.18)
project(pi_calculator LANGUAGES CXX CUDA)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# ---- Source files ----
set(CUDA_SOURCES pi.cu)
set(CXX_SOURCES  pi_opencl.cpp pi_cpu.cpp)

add_executable(pi ${CUDA_SOURCES} ${CXX_SOURCES})

# C++17
set_target_properties(pi PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD  17
)

# Optimization
target_compile_options(pi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# ---- CUDA architectures: sm_50 through sm_100 ----
# Covers Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell.
# Override:  cmake -DCMAKE_CUDA_ARCHITECTURES="89" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_ALL_ARCHS 50 52 60 61 70 75 80 86 89 90)

    # sm_100 (Blackwell) requires CUDA 12.8+
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND _ALL_ARCHS 100)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${_ALL_ARCHS})
endif()
set_target_properties(pi PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ---- NVML (optional) ----
find_library(NVML_LIBRARY
    NAMES nvidia-ml nvml
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        "$ENV{CUDA_PATH}/lib/x64"
        "$ENV{CUDA_PATH}/lib64"
)

if(NVML_LIBRARY)
    message(STATUS "Found NVML: ${NVML_LIBRARY}")
    target_link_libraries(pi PRIVATE ${NVML_LIBRARY})
else()
    message(STATUS "NVML not found — building without GPU utilization stats")
    target_compile_definitions(pi PRIVATE NO_NVML)
endif()

# ---- OpenCL (optional) ----
option(ENABLE_OPENCL "Build with OpenCL support" ON)

if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
        target_link_libraries(pi PRIVATE OpenCL::OpenCL)
        target_compile_definitions(pi PRIVATE HAS_OPENCL)
    else()
        message(STATUS "OpenCL not found — building without OpenCL backend")
    endif()
else()
    message(STATUS "OpenCL disabled by user")
endif()

# ---- OpenMP (optional, for CPU backend) ----
option(ENABLE_OPENMP "Build with OpenMP support for CPU backend" ON)

if(ENABLE_OPENMP)
    # MSVC needs -openmp:llvm for OpenMP 3.1+ task support
    if(MSVC)
        set(OpenMP_CXX_FLAGS "-openmp:llvm")
    endif()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
        target_link_libraries(pi PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(pi PRIVATE HAS_OPENMP)
        # Ensure MSVC uses LLVM OpenMP runtime for task support
        if(MSVC)
            target_compile_options(pi PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-openmp:llvm>)
        endif()
    else()
        message(STATUS "OpenMP not found — CPU backend will run single-threaded")
    endif()
else()
    message(STATUS "OpenMP disabled by user")
endif()

# ---- GMP (required, for CPU Chudnovsky big-number arithmetic) ----
find_library(GMP_LIBRARY NAMES gmp)
if(GMP_LIBRARY)
    message(STATUS "Found GMP: ${GMP_LIBRARY}")
    target_link_libraries(pi PRIVATE ${GMP_LIBRARY})
else()
    message(FATAL_ERROR "GMP not found — required for CPU digit computation")
endif()

# ---- Platform-specific ----
if(WIN32)
    target_compile_definitions(pi PRIVATE _USE_MATH_DEFINES)
endif()

# ---- Link pthreads on Linux (for std::thread in spinner) ----
if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(pi PRIVATE Threads::Threads)
endif()
