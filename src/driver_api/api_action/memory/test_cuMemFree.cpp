#include "memory_tests.h"

TEST_F(CuMemTest, AC_BA_cuMemFree_validParam) {
    CUdeviceptr device_data;
    cuMemAlloc(&device_data, 1024);

    CUresult result = cuMemFree(device_data);

    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_EG_cuMemFree_EdgeCases1) {
    // 之后补充
}

TEST_F(CuMemTest, AC_OT_cuMemFree_leagleLoopInSameContext) {
    for (int i = 0; i < 100; i++) {
        CUdeviceptr device_data;
        cuMemAlloc(&device_data, 1024);
        CUresult result = cuMemFree(device_data);
        EXPECT_EQ(result, CUDA_SUCCESS);
    }
}

TEST_F(CuMemTest, AC_INV_cuMemFree_illeagleLoopInSameContext) {
    CUdeviceptr device_data;
    cuMemAlloc(&device_data, 1024);

    for (int i = 0; i < 100; i++) {
        CUresult result = cuMemFree(device_data);
        if (i == 0) {
            EXPECT_EQ(result, CUDA_SUCCESS);
        } else {
            // EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
            EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
        }
    }
}

TEST_F(CuMemTest, AC_INV_cuMemFree_Unallocatedptr) {
    CUdeviceptr unallocated_ptr = 0xdeedbeef;
    CUresult result = cuMemFree(unallocated_ptr);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemTest, AC_INV_cuMemFree_ullptr) {
    CUdeviceptr null_device_ptr = NULL;
    // 0x00000000 success
    CUresult result = cuMemFree(null_device_ptr);
    EXPECT_EQ(result, CUDA_SUCCESS);
}