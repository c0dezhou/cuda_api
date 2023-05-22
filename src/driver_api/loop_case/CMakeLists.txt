cmake_minimum_required(VERSION 3.8)
project(loop_tests LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories("/usr/local/cuda-11.3/targets/x86_64-linux/include")



# Create the executable
add_executable(loop_tests /data/system/yunfan/cuda_api/src/driver_api/loop_case/test_varify.cpp)

# Link the required libraries
target_link_libraries(loop_tests
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/libcuda.so"
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudart.so"
cuda
pthread)
