cmake_minimum_required(VERSION 3.8)
project(driver_api_tests LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
set( BASE_DIR ${CMAKE_SOURCE_DIR}/../..)
set( COMMON_DIR ${BASE_DIR}/common)
set( DRIVER_INC_DIR ${BASE_DIR}/include/driver_api)
include_directories(${COMMON_DIR})
include_directories(${DRIVER_INC_DIR})

# Locate the Google Test library
enable_testing()
if(NOT GTEST_DIR)
    set( GTEST_DIR ${CMAKE_SOURCE_DIR}/../..)
endif()
 
if( IS_DIRECTORY ${GTEST_DIR} )
    add_subdirectory( ${GTEST_DIR}/third-party/googletest gtestOut)
else()
    message(FATAL_ERROR "INVALID FOLDER 'gtest'=${gtest}" )
endif()

# add the test executable
file(GLOB_RECURSE TEST_SOURCES "test_*.cpp")

# Create the executable
add_executable(driver_api_tests ${COMMON_DIR}/test_utils.cpp ${TEST_SOURCES})

# Link the required libraries
target_link_libraries(driver_api_tests gtest gtest_main 
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/libcuda.so"
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudart.so"
cuda
pthread)
