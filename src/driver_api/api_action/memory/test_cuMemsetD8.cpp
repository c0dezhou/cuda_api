#include "memory_tests.h"

#define INIT_MEMSETD8()                   \
    CUdeviceptr d_p;                      \
    const size_t size = 10 * sizeof(int); \
    cuMemAlloc(&d_p, 10 * sizeof(int));

#define DEL_MEMSETD8() cuMemFree(d_p);

TEST_F(CuMemTest, AC_BA_MemsetD8_BasicApiBehavior) {
    INIT_MEMSETD8();
    unsigned char value = 42;
    cuMemsetD8(d_p, value, size);

    for (size_t i = 0; i < 10; i++) {
        unsigned char value;
        cuMemcpyDtoH(&value, d_p + i, 1);
        EXPECT_EQ(value & 0xff, 42);
    }
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, AC_INV_MemsetD8_Invalidnullptr) {
    INIT_MEMSETD8();
    res = cuMemsetD8(0, 42, size);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMSETD8();
}

// 为啥这个case会影响后面的？
// TEST_F(CuMemTest, AC_INV_MemsetD8_InvalidNegativeSize) {
//     INIT_MEMSETD8();
//     res = cuMemsetD8(d_p, 42, (size_t)(-1));
//     EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
//     DEL_MEMSETD8();
// }

TEST_F(CuMemTest, AC_INV_MemsetD8_InvalidLargeThanAlloc) {
    INIT_MEMSETD8();
    res = cuMemsetD8(d_p, 42, size + 1);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, AC_EG_MemsetD8_ZeroByte) {
    INIT_MEMSETD8();
    CUresult res = cuMemsetD8(d_p, 42, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, AC_EG_MemsetD8_MaxByteSetD8) {
    INIT_MEMSETD8();
    size_t freeMem, totalMem;
    cuMemGetInfo(&freeMem, &totalMem);
    CUdeviceptr bigDevicePtr;
    cuMemAlloc(&bigDevicePtr, freeMem - 1024 * 1024 * 100);
    res = cuMemsetD8(bigDevicePtr, 42, freeMem - 1024 * 1024 * 100);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuMemFree(bigDevicePtr);
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, AC_EG_MemsetD8_UnalignedAllocSetD8) {
    INIT_MEMSETD8();
    res = cuMemsetD8(d_p + 1, 42, size - 2);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, AC_OT_MemsetD8_MultiDeviceSetD8) {
    INIT_MEMSETD8();
    int deviceCount;
    cuDeviceGetCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        CUdevice devicei;
        cuDeviceGet(&devicei, i);
        CUcontext contexti;
        cuCtxCreate(&contexti, 0, devicei);
        cuCtxPushCurrent(contexti);
        CUdeviceptr d_p;
        cuMemAlloc(&d_p, size);
        unsigned char value = 42;
        cuMemsetD8(d_p, value, size);
        for (size_t i = 0; i < 10; i++) {
            unsigned char value;
            cuMemcpyDtoH(&value, d_p + i, 1);
            EXPECT_EQ(value & 0xff, 42);
        }
        cuMemFree(d_p);
        cuCtxPopCurrent(&contexti);
        cuCtxDestroy(contexti);
    }
    DEL_MEMSETD8();
}

TEST_F(CuMemTest, LOOP_MemsetD8_LoopSetD8) {
    unsigned char value = 42;
    const int loopCount = 10;
    INIT_MEMSETD8();
    for (int j = 0; j < loopCount; j++) {
        cuMemsetD8(d_p, value, size);
    }
    for (size_t i = 0; i < 10; i++) {
        unsigned char value;
        cuMemcpyDtoH(&value, d_p + i, sizeof(int));
        EXPECT_EQ(value, 42);
    }
    DEL_MEMSETD8();
}