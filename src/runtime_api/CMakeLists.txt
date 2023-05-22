cmake_minimum_required(VERSION 3.8)
project(rt_event_tests LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set( BASE_DIR ${CMAKE_SOURCE_DIR}/../../..)
include_directories(${BASE_DIR}/common/)

# Locate the Google Test library
enable_testing()
# set( PROJ_DIR ${CMAKE_SOURCE_DIR}/../..)
# add_subdirectory(${PROJ_DIR}/third-party/googletest)

if(NOT NEED_DIR)
    set( NEED_DIR ${CMAKE_SOURCE_DIR}/../..)
endif()
 
if( IS_DIRECTORY ${NEED_DIR} )
    add_subdirectory( ${NEED_DIR}/third-party/googletest gtestOut)
else()
    message(FATAL_ERROR "INVALID FOLDER 'gtest'=${gtest}" )
endif()

# add the test executable
# file(GLOB_RECURSE TEST_SOURCES "test_*.cpp")

# Create the executable
cuda_add_executable(rt_event_tests 
                    api_action/event/test_cudaEventDestroy.cpp
                    api_action/event/test_case_with_kernel.cu)

# Link the required libraries
target_link_libraries(rt_event_tests gtest gtest_main "/usr/local/cuda/lib64/libcudart.so" pthread)
