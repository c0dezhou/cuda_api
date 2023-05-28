#include "memory_tests.h"

#define MAX_MEM 1024*1024*1024 *46 // 46GB
#define INVALID_MEM 0
#define DEAD_HANDLE 0xdeadbeef

TEST_F(CuMemTest, AC_BA_MemAlloc_AllocValidMem) {
    int N = 1024;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N, 1.0f);

    float* d_A;
    cuMemAlloc((CUdeviceptr*)&d_A, bytes);

    cuMemcpyHtoD((CUdeviceptr)d_A, h_A.data(), bytes);

    cuMemFree((CUdeviceptr)d_A);
}

TEST_F(CuMemTest, AC_INV_MemAlloc_Toolargetest) {
    size_t large_size = (size_t)(-1);  // size_t是unsigned的
    CUdeviceptr device_large;
    res = cuMemAlloc(&device_large, large_size);
    EXPECT_EQ(res, CUDA_ERROR_OUT_OF_MEMORY);
}

TEST_F(CuMemTest, AC_INV_MemAlloc_AllocZerobytisize) {
    float* d_A;
    res = cuMemAlloc((CUdeviceptr*)&d_A, 0);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    cuMemFree((CUdeviceptr)d_A);
}

TEST_F(CuMemTest, AC_INV_MemAlloc_AllocInvalidMem) {
    res = cuMemAlloc((CUdeviceptr*)nullptr, (MAX_MEM + 1) * sizeof(float));
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemTest, AC_EG_MemAlloc_AllocMaximumMem) {
    float* d_A;
    res = cuMemAlloc((CUdeviceptr*)&d_A, (MAX_MEM +1)*sizeof(float));
    EXPECT_EQ(res, CUDA_ERROR_OUT_OF_MEMORY);

    cuMemFree((CUdeviceptr)d_A);
}