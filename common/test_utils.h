#pragma once

#include <cuda.h>
#include <gtest/gtest.h>
#include <string.h>
#include <sys/resource.h>
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

void performOperations(int* p, size_t alloc_size);
void initarray(int* p, size_t alloc_size);
void compareOPerations(int* p, size_t alloc_size);
long long getSystemMemoryUsage();
float calculateElapsedTime(const CUevent& start, const CUevent& stop);
void get_random(int* num, int a, int b);
bool __checkError_error(CUresult code,
                        const char* op,
                        const char* file,
                        int line);
bool verifyResult(const std::vector<float>& data,
                  float expectedValue,
                  int startIndex,
                  int endIndex);
bool verifyResult(const std::vector<float>& data, float expectedValue);
bool verifyResult(const std::vector<float>& h_A,
                  const std::vector<float>& h_B,
                  const std::vector<float>& h_C);

void cpuComputation(std::vector<float>& data);

#define checkError(op) __checkError_error((op), #op, __FILE__, __LINE__)
#define MB 1024 * 1024

template <typename T>
void testAllocHost(T value) {
    T* p;
    CUresult res = cuMemAllocHost((void**)&p, sizeof(T));
    EXPECT_EQ(res, CUDA_SUCCESS);
    *p = value;
    EXPECT_EQ(*p, value);
    cuMemFreeHost(p);
}

template <typename F>
double measureTime(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
void test_memcpy_htod(int if_async) {
    size_t num_elements = 10;
    std::vector<T> h_A(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        h_A[i] = static_cast<T>(i);
    }

    CUdeviceptr d_A;
    cuMemAlloc(&d_A, num_elements * sizeof(T));

    if (if_async == 0) {
        cuMemcpyHtoD(d_A, h_A.data(), num_elements * sizeof(T));
    } else {
        // 暂时用默认流
        cuMemcpyHtoDAsync(d_A, h_A.data(), num_elements * sizeof(T), 0);
    }

    std::vector<T> h_A_copy(num_elements);
    cuMemcpyDtoH(h_A_copy.data(), d_A, num_elements * sizeof(T));
    cuMemFree(d_A);

    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(h_A[i], h_A_copy[i]);
    }
}