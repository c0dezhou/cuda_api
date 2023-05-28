#include "memory_tests.h"

#define INIT_GETMEMINFO() \
    size_t free;          \
    size_t total;         \
    size_t free_before;   \
    size_t free_after;    \
    CUdeviceptr d_ptr;    \
    CUresult result;

TEST_F(CuMemTest, AC_BA_MemGetInfo_BasicBehavior) {
    INIT_GETMEMINFO();
    cuMemGetInfo(&free, &total);

    EXPECT_GT(free, 0); // >
    EXPECT_GT(total, 0);

    EXPECT_LE(free, total);
}

TEST_F(CuMemTest, AC_INV_MemGetInfo_InvaliFreePtr) {
    INIT_GETMEMINFO();
    result = cuMemGetInfo(NULL, &total);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_INV_MemGetInfo_InvaliTotalPtr) {
    INIT_GETMEMINFO();

    result = cuMemGetInfo(&free, NULL);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_INV_MemGetInfo_InvaliNUllptr) {
    INIT_GETMEMINFO();
    result = cuMemGetInfo(NULL, NULL);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuMemTest, AC_EG_MemGetInfo_MaxGetMemInfo) {
    INIT_GETMEMINFO();

    cuMemGetInfo(&free_before, &total);

    // 分配最大的设备内存
    result = cuMemAlloc(&d_ptr, free_before);
    EXPECT_EQ(result, CUDA_ERROR_OUT_OF_MEMORY);

    result = cuMemAlloc(&d_ptr, free_before - 1024*1024*1024);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuMemGetInfo(&free_after, &total);
    EXPECT_NEAR(free_after, 0, 1024*1024*1024);

    cuMemFree(d_ptr);
}

TEST_F(CuMemTest, AC_OT_MemGetInfo_MultiContextGetMemInfo) {
    CUcontext context2;

    size_t free2,free1,total1,total2;

    cuMemGetInfo(&free1, &total1);

    cuCtxCreate(&context2, 0, device);
    cuCtxSetCurrent(context2);

    cuMemGetInfo(&free2, &total2);

    EXPECT_NEAR(free2, free1, 1024*1024*600);
    EXPECT_EQ(total2, total1);

    cuCtxDestroy(context2);
}