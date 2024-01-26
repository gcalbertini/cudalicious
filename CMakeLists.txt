# https://stackoverflow.com/questions/67888279/debugging-cuda-kernels-with-vs-code
# https://stackoverflow.com/questions/14148933/libcmt-conflicts-with-use-of-other-libs-unresolved-external-symbols
# https://stackoverflow.com/questions/757418/should-i-compile-with-md-or-mt

cmake_minimum_required(VERSION 3.18)

project(MultipleCUDAExamples CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

message(STATUS  "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS  "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

# Add common source file(s)
#file(GLOB_RECURSE COMMON_SOURCES "${CMAKE_SOURCE_DIR}/src/common.cpp")
#file(GLOB_RECURSE COMMON_HEADERS "${CMAKE_SOURCE_DIR}/include/common.h")

# Include directories
include_directories("${CMAKE_SOURCE_DIR}/include")

# Function to add an executable for a CUDA source file
function(add_cuda_executable CUDA_SOURCE)
    get_filename_component(EXE_NAME ${CUDA_SOURCE} NAME_WE)
    add_executable(${EXE_NAME} ${CUDA_SOURCE} ${COMMON_SOURCES} ${COMMON_HEADERS})
    
    # Use -G option with nvcc for debugging information in Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
        message(STATUS  "Building in Debug mode for ${EXE_NAME}")
    else()
        # Print a message indicating Release mode
        message(STATUS  "Building in Release mode for ${EXE_NAME}")
    endif()

    # Specify CUDA standard
    target_compile_features(${EXE_NAME} PRIVATE cuda_std_17)

    # Set the CUDA architectures explicitly; here have the RTX 2080, and Turing has 75 SMs
    set_target_properties(${EXE_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
endfunction()

# Find all CUDA source files
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")

# Create an executable for each CUDA file
foreach(CUDA_SOURCE ${CUDA_SOURCES})
    add_cuda_executable(${CUDA_SOURCE})
endforeach()

# For Debug/Release build
#cmake -DCMAKE_BUILD_TYPE=Debug or not
#cmake --build . --config Debug/Release
