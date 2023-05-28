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
    // TODO: 解决
    // 设备指针和主机指针的对齐取决于所使用的特定 GPU 架构
    // 一些较旧的体系结构要求内存与特定操作的特定边界对齐
    INIT_MEMH2D();
    res = cuMemcpyHtoD(d_p + 1, h_p + 1, size - 2);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
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
    // TODO：解决
    CUdeviceptr d_p;
    int* h_p;
    const size_t size = 1024;
    CUresult res;
    res = cuMemAllocHost((void**)&h_p, size);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuMemAlloc(&d_p, size);
    EXPECT_EQ(res, CUDA_SUCCESS);
    int deviceCount;
    cuDeviceGetCount(&deviceCount);

    for (int i = 0; i < size / sizeof(int); i++) {
        h_p[i] = i;
    }

    for (int i = 0; i < deviceCount; i++) {
        CUdevice devicei;
        cuDeviceGet(&devicei, i);
        CUcontext contexti;
        res = cuCtxCreate(&contexti, 0, devicei);
        EXPECT_EQ(res, CUDA_SUCCESS);
        cuCtxPushCurrent(contexti);
        CUdeviceptr d_pi;
        res = cuMemAlloc(&d_pi, size);
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuMemcpyHtoD(d_pi, h_p, size);
        EXPECT_EQ(res, CUDA_SUCCESS);
        for (size_t i = 0; i < size / sizeof(int); i++) {
            int value;
            cuMemcpyDtoH(&value, d_pi + i * sizeof(int), sizeof(int));
            EXPECT_EQ(value, i);
        }
        cuMemFree(d_pi);
        cuCtxPopCurrent(nullptr);
        cuCtxDestroy(contexti);
    }
    cuMemFreeHost(h_p);
    cuMemFree(d_p);
}
