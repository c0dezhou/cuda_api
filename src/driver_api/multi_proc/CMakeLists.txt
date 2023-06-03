cmake_minimum_required(VERSION 3.8)
project(proc_test_runner LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(LIBS
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/libcuda.so"
"/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudart.so"
cuda
pthread)

add_executable (md_fork_large_data multi_dev/md_fork_large_data.cpp)
target_link_libraries(md_fork_large_data ${LIBS})

add_executable (md_fork_multi multi_dev/md_fork_multi.cpp)
target_link_libraries(md_fork_multi ${LIBS})

add_executable (md_fork multi_dev/md_fork.cpp)
target_link_libraries(md_fork ${LIBS})

add_executable (md_spwan_child multi_dev/md_spwan_child.cpp)
target_link_libraries(md_spwan_child ${LIBS})

add_executable (md_spwan_parent multi_dev/md_spwan_parent.cpp)
target_link_libraries(md_spwan_parent ${LIBS})

add_executable (fork_large_data single_dev/fork_large_data.cpp)
target_link_libraries(fork_large_data ${LIBS})

add_executable (fork single_dev/fork.cpp)
target_link_libraries(fork ${LIBS})

add_executable (spwan_child single_dev/spwan_child.cpp)
target_link_libraries(spwan_child ${LIBS})

add_executable (spwan_parent single_dev/spwan_parent.cpp)
target_link_libraries(spwan_parent ${LIBS})