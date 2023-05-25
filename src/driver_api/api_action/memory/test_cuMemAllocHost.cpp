#include "memory_tests.h"

TEST_F(CuMemTest, AC_BA_MemAllocHost_BasicFuncDiffdt) {
    testAllocHost<int>(12345);

    testAllocHost<double>(12345.6789);

    const int arr_size = 1024;
    float* pfloat_arr;
    CUresult res =
        cuMemAllocHost((void**)&pfloat_arr, arr_size * sizeof(float));
    EXPECT_EQ(res, CUDA_SUCCESS);
    for (int i = 0; i < arr_size; i++) {
        pfloat_arr[i] = static_cast<float>(i);
        EXPECT_FLOAT_EQ(pfloat_arr[i], static_cast<float>(i));
    }
    cuMemFreeHost(pfloat_arr);
}

TEST_F(CuMemTest, AC_INV_MemAllocHost_ZeroByte) {
    int* pZeroBytes;
    CUresult res = cuMemAllocHost((void**)&pZeroBytes, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_INV_MemAllocHost_Largesize) {
    int* pNegativeSize;
    res = cuMemAllocHost((void**)&pNegativeSize, -1);
    EXPECT_EQ(res, CUDA_ERROR_OUT_OF_MEMORY);
}

TEST_F(CuMemTest, AC_INV_MemAllocHost_Nullptr) {
    // TODO：待确认
    const int* pInvalidPtr = nullptr;
    CUresult res = cuMemAllocHost((void**)&pInvalidPtr, sizeof(int));
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemTest, AC_INV_MemAllocHost_Overbyte) {
    // TODO：待确认
    int* pIncorrectSize;
    // 不匹配bytesize
    res = cuMemAllocHost((void**)&pIncorrectSize, sizeof(float) + 1);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

// data integrity
TEST_F(CuMemTest, AC_OT_MemAllocHost_DataIntegrity) {
    const size_t alloc_size = 1024 * 1024;  // 1 MB

    int* p;
    CUresult res = cuMemAllocHost((void**)&p, alloc_size * sizeof(int));

    if (res == CUDA_SUCCESS) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        for (size_t i = 0; i < alloc_size; i++) {
            p[i] = dist(gen);
        }

        for (size_t i = 0; i < alloc_size; i++) {
            int expected = p[i];
            int actual = p[i];
            EXPECT_EQ(actual, expected)
                << "Data corruption at " << i;
        }
        cuMemFreeHost(p);
    }
}

TEST_F(CuMemTest, AC_OT_MemAllocHost_MemoryDeallocation) {
    const size_t alloc_size = 1024 * 1024;  // 1 MB

    long long sys_mem_usage = getSystemMemoryUsage();

    int* p;
    CUresult res = cuMemAllocHost((void**)&p, alloc_size * sizeof(int));

    if (res == CUDA_SUCCESS) {
        performOperations(p, alloc_size);

        cuMemFreeHost(p);
        long long curr_mem_usage = getSystemMemoryUsage();
        EXPECT_EQ(curr_mem_usage, sys_mem_usage);
    }
}