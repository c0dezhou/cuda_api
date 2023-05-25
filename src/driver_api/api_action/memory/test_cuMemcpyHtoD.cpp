#include "memory_tests.h"

#define INIT_MEMH2D()           \
    CUdeviceptr d_p;            \
    int* h_p;                  \
    const size_t size = 1024;   \
    cuMemAllocHost((void**)&h_p, size); \
    cuMemAlloc(&d_p, size);

#define DEL_MEMH2D()    \
    cuMemFreeHost(h_p); \
    cuMemFree(d_p);

TEST_F(CuMemTest, AC_BA_MemcpyHtoD_BasicFunctionality) {
    test_memcpy_htod<int8_t>(0);
    test_memcpy_htod<uint8_t>(0);
    test_memcpy_htod<int16_t>(0);
    test_memcpy_htod<uint16_t>(0);
    test_memcpy_htod<int32_t>(0);
    test_memcpy_htod<uint32_t>(0);
    test_memcpy_htod<int64_t>(0);
    test_memcpy_htod<uint64_t>(0);
    test_memcpy_htod<float>(0);
    test_memcpy_htod<double>(0);
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoD_InvalidNullhostptr) {
    INIT_MEMH2D();
    CUresult res = cuMemcpyHtoD(d_p, nullptr, size);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoD_InvalidNulldeviceptr) {
    INIT_MEMH2D();
    res = cuMemcpyHtoD(0, h_p, size);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoD_InvalidNegativeSize) {
    INIT_MEMH2D();
    res = cuMemcpyHtoD(d_p, h_p, -1);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoD_InvalidLargeThanAlloc) {
    INIT_MEMH2D();
    res = cuMemcpyHtoD(d_p, h_p, size + 1);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoD_ZeroByte) {
    INIT_MEMH2D();
    CUresult res = cuMemcpyHtoD(d_p, h_p, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoD_MaxByte) {
    INIT_MEMH2D();
    size_t freeMem, totalMem;
    cuMemGetInfo(&freeMem, &totalMem);
    void* bigh_p;
    CUdeviceptr bigd_p;
    res = cuMemAllocHost(&bigh_p, freeMem - 1014 * 1024 * 100);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuMemAlloc(&bigd_p, freeMem - 1014 * 1024 * 100);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuMemcpyHtoD(bigd_p, bigh_p, freeMem);
    EXPECT_EQ(res, CUDA_SUCCESS) << freeMem;
    cuMemFreeHost(bigh_p);
    cuMemFree(bigd_p);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoD_UnalignedAddr) {
    // TODO：待确认
    INIT_MEMH2D();
    res = cuMemcpyHtoD(d_p + 1, h_p + 1, size - 2);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_SA_MemcpyHtoD_SyncBehavior) {
    INIT_MEMH2D();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    cuMemcpyHtoD(d_p, h_p, size);

    res = cuStreamQuery(stream);
    EXPECT_EQ(res, CUDA_SUCCESS);

    cuStreamDestroy(stream);
    DEL_MEMH2D();
}

TEST_F(CuMemTest, AC_OT_MemcpyHtoD_MultiDevice) {
    // TODO：待确认
    GTEST_SKIP();
    INIT_MEMH2D();
    int deviceCount;
    cuDeviceGetCount(&deviceCount);

    for (int i = 0; i < size/sizeof(int); i++){
        h_p[i] = i;
    }

    for (int i = 0; i < deviceCount - 1; i++) {
        CUdevice devicei;
        cuDeviceGet(&devicei, i);
        CUcontext contexti;
        cuCtxCreate(&contexti, 0, devicei);
        cuCtxPushCurrent(contexti);
        CUdeviceptr d_pi;
        cuMemAlloc(&d_pi, size);
        cuMemcpyHtoD(d_pi, h_p, size);
        for (size_t i = 0; i < size / sizeof(int); i++) {
            int value;
            cuMemcpyDtoH(&value, d_pi + i, size);
            EXPECT_EQ(value, i);
        }
        cuMemFree(d_pi);
        cuCtxPopCurrent(&contexti);
        cuCtxDestroy(contexti);
    }
    DEL_MEMH2D();
}
