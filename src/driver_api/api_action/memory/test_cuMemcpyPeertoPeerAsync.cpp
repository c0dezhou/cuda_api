#include <gtest/gtest.h>
#include "cuda_runtime.h"
#include "memory_tests.h"

class CuMemcpyPeerAsyncTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cuInit(0);
        cuDeviceGetCount(&device_count);
        if (device_count < 2) {
            GTEST_SKIP();
        }
        cuStreamCreate(&stream, 0);
        cuDeviceGet(&device1, 0);
        cuDeviceGet(&device2, 1);
        cuCtxCreate(&context1, 0, device1);
        cuCtxCreate(&context2, 0, device2);

        cuCtxSetCurrent(context1);
        cuMemAlloc(&dptr1, size * sizeof(int));

        cuCtxSetCurrent(context2);
        cuMemAlloc(&dptr2, size * sizeof(int));
    }

    void TearDown() override {
        cuCtxSetCurrent(context1);
        cuMemFree(dptr1);

        cuCtxSetCurrent(context1);
        cuMemFree(dptr2);

        cuCtxDestroy(context1);
        cuCtxDestroy(context2);
        cuStreamDestroy(stream);
    }

    const size_t size = 1024;
    int device_count;
    CUdeviceptr dptr1;
    CUdeviceptr dptr2;
    CUstream stream;
    CUdevice device1, device2;
    CUcontext context1, context2;
};

TEST_F(CuMemcpyPeerAsyncTest, AC_BA_MemcpyPeerAsync_BasicBehavior) {
    // int access = -1;
    // cuDeviceCanAccessPeer(&access, device1, device2);
    // std::cout << access <<std::endl;
    int hptr[size];
    for (int i = 0; i < size; i++) {
        hptr[i] = i;
    }
    cuCtxSetCurrent(context1);
    cuMemcpyHtoD(dptr1, hptr, size * sizeof(int));

    cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, size * sizeof(int),
                      stream);

    cuStreamSynchronize(stream);
    cudaDeviceSynchronize();

    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(hptr[i], i);
    }
}

TEST_F(CuMemcpyPeerAsyncTest, AC_INV_MemcpyPeerAsync_InvalidDevice) {
    GTEST_SKIP();  // due to kasi
    CUdevice device3, device4;
    CUcontext context3, context4;
    CUdeviceptr dptr3, dptr4;

    cuDeviceGet(&device3, 100);
    cuDeviceGet(&device4, -1);
    cuCtxCreate(&context1, 0, device3);
    cuCtxCreate(&context2, 0, device4);

    cuCtxSetCurrent(context3);
    cuMemAlloc(&dptr3, size * sizeof(int));

    cuCtxSetCurrent(context4);
    cuMemAlloc(&dptr4, size * sizeof(int));

    EXPECT_EQ(cuMemcpyPeerAsync(dptr2, context3, dptr1, context1,
                                size * sizeof(int), stream),
              CUDA_ERROR_INVALID_DEVICE);
    EXPECT_EQ(cuMemcpyPeerAsync(dptr1, context1, dptr2, context4,
                                size * sizeof(int), stream),
              CUDA_ERROR_INVALID_DEVICE);
}

TEST_F(CuMemcpyPeerAsyncTest, AC_INV_MemcpyPeerAsync_Invaliddeviceptr) {
    EXPECT_EQ(cuMemcpyPeerAsync(0, context2, dptr1, context1,
                                size * sizeof(int), stream),
              CUDA_ERROR_INVALID_VALUE);
    EXPECT_EQ(cuMemcpyPeerAsync(dptr2, context2, 0, context1,
                                size * sizeof(int), stream),
              CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemcpyPeerAsyncTest, AC_INV_MemcpyPeerAsync_InvalidSameDevice) {
    EXPECT_EQ(cuMemcpyPeerAsync(dptr1, context1, dptr1, context1,
                                size * sizeof(int), stream),
              CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemcpyPeerAsyncTest, AC_EG_MemcpyPeerAsync_EdgeCasesZeroByte) {
    EXPECT_EQ(cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, 0, stream),
              CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuMemcpyPeerAsyncTest, AC_EG_MemcpyPeerAsync_EdgeCasesWalkOne) {
    int hptr[size];
    for (int i = 0; i < size; i++) {
        hptr[i] = i;
    }
    cuCtxSetCurrent(context1);
    cuMemcpyHtoD(dptr1, hptr, size * sizeof(int));

    cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, 1, stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    EXPECT_EQ(hptr[0], 0);

    for (int i = 1; i < size; i++) {
        EXPECT_EQ(hptr[i], 0);
    }

    cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, size * sizeof(int),
                      stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(hptr[i], i);
    }
}

// 异步的，是否可以与其他操作重叠
TEST_F(CuMemcpyPeerAsyncTest, AC_EG_MemcpyPeerAsync_AsyncBehavior) {
    int hptr[size];
    for (int i = 0; i < size; i++) {
        hptr[i] = i;
    }
    cuCtxSetCurrent(context1);
    cuMemcpyHtoD(dptr1, hptr, size * sizeof(int));

    CUmodule cuModule;
    CUfunction cuFunction;

    cuModuleLoad(&cuModule,
                 "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                 "cuda_kernel.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "_Z18arraySelfIncrementPii");

    cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, size * sizeof(int),
                      stream);

    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    int sharedMemBytes = 256;
    int n = size;
    void* kernelParams[] = {(void*)dptr1, (void*)&n};
    cuLaunchKernel(cuFunction, gridDimX, gridDimY, gridDimZ, blockDimX,
                   blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams,
                   nullptr);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(hptr[i], i);
    }

    cuMemcpyDtoH(hptr, dptr1, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(hptr[i], i + 1);
    }
}

TEST_F(CuMemcpyPeerAsyncTest, AC_EG_MemcpyPeerAsync_RepeatedCalls) {
    int hptr[size];
    for (int i = 0; i < size; i++) {
        hptr[i] = i;
    }
    cuCtxSetCurrent(context1);
    cuMemcpyHtoD(dptr1, hptr, size * sizeof(int));

    for (int i = 0; i < 10; i++) {
        cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, size * sizeof(int),
                          stream);
        cuMemcpyPeerAsync(dptr1, context1, dptr2, context2, size * sizeof(int),
                          stream);
    }

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(hptr, dptr1, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(hptr[i], i);
    }
}

TEST_F(CuMemcpyPeerAsyncTest,
       AC_EG_MemcpyPeerAsync_DifferentSizesAndAlignments) {
    int hptr[size];
    for (int i = 0; i < size; i++) {
        hptr[i] = i;
    }
    cuCtxSetCurrent(context1);
    cuMemcpyHtoD(dptr1, hptr, size * sizeof(int));

    // 复制除了最后一个字节的所有内存
    cuMemcpyPeerAsync(dptr2, context2, dptr1, context1, size - 1, stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    for (int i = 0; i < size - 1; i++) {
        EXPECT_EQ(hptr[i], i);
    }
    EXPECT_EQ(hptr[size - 1], 0);

    // 复制不对齐的内存
    cuMemcpyPeerAsync(dptr2 + 3, context2, dptr1 + 3, context1, size - 6,
                      stream);

    cuStreamSynchronize(stream);

    // 前后三个字节是否为0
    cuMemcpyDtoH(hptr, dptr2, size * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (i == 0 || i == size - 1) {
            EXPECT_EQ(hptr[i], 0);
        } else {
            EXPECT_EQ(hptr[i], (i << 8) + (i - 1));
        }
    }
}