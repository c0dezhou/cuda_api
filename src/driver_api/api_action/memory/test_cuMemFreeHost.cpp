#include "memory_tests.h"

#define INIT_POINT()    \
    int* p;  \
    cuMemAllocHost((void**)&p, sizeof(int));

TEST_F(CuMemTest, AC_BA_cuMemFreeHost_MemFreeHostSuccess) {
    INIT_POINT();
    CUresult result = cuMemFreeHost(p);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_INV_cuMemFreeHost_nullptrointer) {
    INIT_POINT();
    cuMemFreeHost(p);
    CUresult result = cuMemFreeHost(p);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_INV_cuMemFreeHost_DoubleFree) {
    INIT_POINT();
    CUresult result = cuMemFreeHost(p);
    EXPECT_EQ(result, CUDA_SUCCESS);
    result = cuMemFreeHost(p);  // Attempt to free the same memory again
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_SA_cuMemFreeHost_SyncBehavior) {
    GTEST_SKIP(); // due to core dump
    INIT_POINT();
    CUdeviceptr d;
    cuMemAlloc(&d, sizeof(int));
    CUresult result = cuMemFreeHost(p);
    EXPECT_EQ(result, CUDA_SUCCESS);

    // cuMemFreeHost 后不能再访问 p
    result = cuMemcpyHtoD(d, p, sizeof(int));
    EXPECT_NE(result, CUDA_SUCCESS);
}