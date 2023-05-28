#include "memory_tests.h"

#define INIT_MEMH2DAsync()      \
    CUstream stream;            \
    cuStreamCreate(&stream, 0); \
    CUdeviceptr d_p;            \
    void* h_p;                  \
    const size_t size = 1024;   \
    cuMemAllocHost(&h_p, size); \
    cuMemAlloc(&d_p, size);

#define DEL_MEMH2DAsync()    \
    cuStreamDestroy(stream); \
    cuMemFreeHost(h_p);      \
    cuMemFree(d_p);

TEST_F(CuMemTest, AC_BA_MemcpyHtoDAsync_BasicFunctionality) {
    test_memcpy_htod<int8_t>(1);
    test_memcpy_htod<uint8_t>(1);
    test_memcpy_htod<int16_t>(1);
    test_memcpy_htod<uint16_t>(1);
    test_memcpy_htod<int32_t>(1);
    test_memcpy_htod<uint32_t>(1);
    test_memcpy_htod<int64_t>(1);
    test_memcpy_htod<uint64_t>(1);
    test_memcpy_htod<float>(1);
    test_memcpy_htod<double>(1);
}

TEST_F(CuMemTest, AC_BA_MemcpyHtoDAsync_InvalidNullhostptr) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(d_p, nullptr, size, stream);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoDAsync_InvalidNulldeviceptr) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(0, h_p, size, stream);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoDAsync_InvalidNegativeSize) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(d_p, h_p, -1, stream);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoDAsync_InvalidLargeThanAlloc) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(d_p, h_p, size + 1, stream);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyHtoDAsync_InvalidStream) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(d_p, h_p, size, (CUstream)(-1));
    EXPECT_EQ(res, CUDA_ERROR_INVALID_HANDLE);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoDAsync_ZeroByte) {
    INIT_MEMH2DAsync();
    CUresult res = cuMemcpyHtoDAsync(d_p, h_p, 0, stream);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoDAsync_MaxByte) {
    INIT_MEMH2DAsync();
    size_t freeMem, totalMem;
    cuMemGetInfo(&freeMem, &totalMem);
    void* bigh_p;
    CUdeviceptr bigd_p;
    cuMemAllocHost(&bigh_p, freeMem);
    cuMemAlloc(&bigd_p, freeMem);
    res = cuMemcpyHtoDAsync(bigd_p, bigh_p, freeMem, stream);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuMemFreeHost(bigh_p);
    cuMemFree(bigd_p);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyHtoDAsync_UnalignedAddr) {
    INIT_MEMH2DAsync();
    res = cuMemcpyHtoDAsync(d_p + 1, h_p + 1, size - 2, stream);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMH2DAsync();
}

TEST_F(CuMemTest, AC_SA_MemcpyHtoDAsync_AsyncBehavior) {
    // TODO：解决
    // GTEST_SKIP();  // due to core dump
    CUstream stream1, stream2;
    CUdeviceptr d_p;
    char* h_p;
    const size_t size = 1024;

    cuStreamCreate(&stream1, 0);
    cuStreamCreate(&stream2, 0);

    cuMemAllocHost((void**)&h_p, size);
    cuMemAlloc(&d_p, size);

    for (size_t i = 0; i < size; i++) {
        h_p[i] = i % 256;
    }

    cuMemcpyHtoDAsync(d_p, h_p, size, stream1);

    cuStreamSynchronize(stream1);

    cuMemcpyDtoHAsync(h_p, d_p, size, stream2);

    cuStreamSynchronize(stream2);

    for (size_t i = 0; i < size; i++) {
        char value = h_p[i];
        EXPECT_EQ(value & 0xff, i % 256);
    }

    cuMemFreeHost(h_p);
    cuMemFree(d_p);

    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
}

TEST_F(CuMemTest, AC_OT_MemcpyHtoDAsync_MultiDevice) {
    // TODO：解决
    CUstream stream;
    CUdeviceptr d_p;
    char* h_p;
    const size_t size = 1024;

    cuStreamCreate(&stream, 0);

    cuMemAllocHost((void**)&h_p, size);
    cuMemAlloc(&d_p, size);

    for (size_t i = 0; i < size; i++) {
        h_p[i] = i % 256;
    }

    int deviceCount;
    cuDeviceGetCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        CUdevice device;
        cuDeviceGet(&device, i);
        CUcontext contexti;
        cuCtxCreate(&contexti, 0, device);
        cuCtxPushCurrent(contexti);

        CUdeviceptr d_pi;
        cuMemAlloc(&d_pi, size);

        CUstream streami;
        cuStreamCreate(&streami, 0);

        cuMemcpyHtoDAsync(d_pi, h_p, size, streami);

        cuStreamSynchronize(streami);

        for (size_t i = 0; i < size; i++) {
            char value;
            cuMemcpyDtoH(&value, d_pi + i, sizeof(char));
            EXPECT_EQ(value, h_p[i]);
        }

        cuStreamDestroy(streami);
        cuMemFree(d_pi);
        cuCtxPopCurrent(&contexti);
        cuCtxDestroy(contexti);
    }

    cuStreamDestroy(stream);
    cuMemFreeHost(h_p);
    cuMemFree(d_p);
}

TEST_F(CuMemTest, AC_OT_MemcpyHtoDAsync_LoopH2DAsync) {
    INIT_MEMH2DAsync();
    for (size_t i = 0; i < size/sizeof(char); i++) {
        ((char*)h_p)[i] = i % 256;
    }
    const int loopCount = 10;
    for (int j = 0; j < loopCount; j++) {
        cuMemcpyHtoDAsync(d_p, h_p, size, stream);
    }
    cuStreamSynchronize(stream);
    for (size_t i = 0; i < size/sizeof(char); i++) {
        char value;
        cuMemcpyDtoH(&value, d_p + i, 1);
        // & 一下0xff，不然是字符串'0xxx'
        EXPECT_EQ(value & 0xff, i % 256); 
    }
    DEL_MEMH2DAsync();
}