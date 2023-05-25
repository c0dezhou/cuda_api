#include "memory_tests.h"

#define INIT_MEMSETD8Async()   \
    CUdeviceptr d_p;        \
    const size_t size = 10*sizeof(unsigned char);   \
    cuMemAlloc(&d_p, size); \
    unsigned char h_ptr_[10];

#define DEL_MEMSETD8Async() \
    cuMemFree(d_p);

TEST_F(CuMemTest, AC_INV_MemsetD8Async_BasicApiBehavior) {
    INIT_MEMSETD8Async();
    unsigned char uc = 0xFF;
    cuMemsetD8Async(d_p, uc, size, 0);

    cuStreamSynchronize(0);

    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size/sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i] & 0xff, uc);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_INV_MemsetD8Async_Invalidnullptr) {
    INIT_MEMSETD8Async();
    res = cuMemsetD8Async(0, 42, size, 0);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMSETD8Async();
}

// TEST_F(CuMemTest, InvalidNegativeSizeSetD8Async) {
//     INIT_MEMSETD8Async();
//     res = cuMemsetD8Async(d_p, 42, -1, 0);
//     EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
//     DEL_MEMSETD8Async();
// }

TEST_F(CuMemTest, AC_INV_MemsetD8Async_InvalidLargeThanAlloc) {
    INIT_MEMSETD8Async();
    res = cuMemsetD8Async(d_p, 42, size + 1, 0);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_INV_MemsetD8Async_Invalidstream) {
    INIT_MEMSETD8Async();
    // CUstream stream_invalid;
    // res = cuMemsetD8Async(d_p, 42, size, stream_invalid);
    // EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    CUstream stream_invalid;
    res = cuMemsetD8Async(d_p, 42, size, (CUstream)(-1));
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_ZeroByte) {
    INIT_MEMSETD8Async();
    CUresult res = cuMemsetD8Async(d_p, 42, 0, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_Uc0x00) {
    INIT_MEMSETD8Async();
    CUresult res = cuMemsetD8Async(d_p, 0x00, size, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i], 0x00);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_Uc0xff) {
    INIT_MEMSETD8Async();
    CUresult res = cuMemsetD8Async(d_p, 0xff, size, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i], 0xff);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_Uc0x80) {
    INIT_MEMSETD8Async();
    // 0x80是最高位为1的最小值 
    CUresult res = cuMemsetD8Async(d_p, 0x80, size, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i], 0x80);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_Uc0x7f) {
    INIT_MEMSETD8Async();
    // 0x7f最高位为0的最大值
    CUresult res = cuMemsetD8Async(d_p, 0x7f, size, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i], 0x7f);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_MaxByte) {
    INIT_MEMSETD8Async();
    size_t freeMem, totalMem;
    cuMemGetInfo(&freeMem, &totalMem);
    CUdeviceptr bigDevicePtr;
    cuMemAlloc(&bigDevicePtr, freeMem - 1024 * 1024 * 100);
    res = cuMemsetD8Async(bigDevicePtr, 42, freeMem - 1024 * 1024 * 100, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    cuMemFree(bigDevicePtr);
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_EG_MemsetD8Async_UnalignedAlloc) {
    INIT_MEMSETD8Async();
    res = cuMemsetD8Async(d_p + 1, 42, size - 2, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_OT_MemsetD8Async_MultiDevice) {
    INIT_MEMSETD8Async();
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
        cuMemsetD8Async(d_p, value, size, 0);
        for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
            unsigned char value;
            cuMemcpyDtoH(&value, d_p + i, 1);
            EXPECT_EQ(value, 42);
        }
        cuMemFree(d_p);
        cuCtxPopCurrent(&contexti);
        cuCtxDestroy(contexti);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, MemsetD8Async_AsyncBehaviorSetD8Async) {
    // TODO: 解决
    // 在同步发出cuMemsetD8Async操作的流之前，您正在执行设备到主机的内存复制。此操作是异步的，这意味着它立即将控制返回给CPU并并行地完成操作。因此，当在cuMemsetD8Async之后立即执行cuMemcpyDtoH时，有可能memset操作尚未完成，从而导致您复制尚未设置的数据。
    CUdeviceptr d_p;        
    const size_t size = 10*sizeof(unsigned char);   
    cuMemAlloc(&d_p, size); 
    unsigned char h_ptr_[10];
    unsigned char uc = 0xFF;
    cuMemsetD8Async(d_p, uc, size, 0);

    cuStreamSynchronize(0);  // Synchronize the stream right after cuMemsetD8Async

    cuMemcpyDtoH(h_ptr_, d_p, size);
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i] & 0xff, uc);
    }

    cuMemFree(d_p);
}


TEST_F(CuMemTest, LOOP_LoopSetD8Async) {
    unsigned char value = 42;
    const int loopCount = 10;
    INIT_MEMSETD8Async();
    for (int j = 0; j < loopCount; j++) {
        cuMemsetD8Async(d_p, value, size, 0);
    }
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        unsigned char value;
        cuMemcpyDtoH(&value, d_p + i, 1);
        EXPECT_EQ(value, 42);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_OT_MemsetD8Async_RepeatedCallSetD8Async) {
    INIT_MEMSETD8Async();
    unsigned char uc = 0xFF;
    cuMemsetD8Async(d_p, uc, size, 0);
    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);

    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i] & 0xff, uc);
    }

    uc = 0x00;
    cuMemsetD8Async(d_p, uc, size, 0);

    cuStreamSynchronize(0);
    cuMemcpyDtoH(h_ptr_, d_p, size);

    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i] & 0xff, uc);
    }
    DEL_MEMSETD8Async();
}

TEST_F(CuMemTest, AC_OT_MemsetD8Async_overflowSetD8Async) {
    // TODO: 解决
    // 流销毁：在代码的最后一行中，试图销毁默认流（流 0）。 根据 CUDA 编程指南，不应破坏默认流。 调用 cuStreamDestroy(0); 可能会导致问题。
    CUdeviceptr d_p;        
    const size_t size = 10*sizeof(unsigned char);   
    cuMemAlloc(&d_p, size); 
    unsigned char h_ptr_[10];

    unsigned char uc = 0xFF;
    cuMemsetD8Async(d_p, uc, size, 0);

    CUmodule cuModule;
    CUfunction cuFunction;

    cuModuleLoad(&cuModule,
                 "/data/system/zz/cuda_api/common/cuda_kernel/"
                 "cuda_kernel.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "_Z18arraySelfIncrementPii");

    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    int sharedMemBytes = 256;
    void* kernelParams[] = {(void*)d_p, (void*)&size};
    cuLaunchKernel(cuFunction, gridDimX, gridDimY, gridDimZ, blockDimX,
                   blockDimY, blockDimZ, sharedMemBytes, 0, kernelParams,
                   nullptr);

    cuStreamSynchronize(0);

    cuMemcpyDtoH(h_ptr_, d_p, size);

    // 0xFF + 1 = 0x00（溢出）
    for (size_t i = 0; i < size / sizeof(unsigned char); i++) {
        EXPECT_EQ(h_ptr_[i], 0x00);
    }

    // cuStreamDestroy(0);
    cuMemFree(d_p);
}