// Resource Limitation
// 旨在测试cuda driver
// api的资源限制，例如内存分配，线程数，流数，事件数等。使用了一些随机的参数和循环

#include "test_utils.h"


TEST(CudaDriverApiTest, MemoryAllocation) {
    int N = 1000;
    int M = 1024;
    CUdevice device;
    CUcontext context;
    CUdeviceptr devPtr[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    for (int i = 0; i < N; i++) {
        checkError(cuMemAlloc(&devPtr[i], M));
        checkError(cuMemFree(devPtr[i]));
    }

    checkError(cuCtxDestroy(context));
}


TEST(CudaDriverApiTest, ThreadCreation) {
    int N = 1000;  // threads per block
    int M = 1000;  // blocks per grid
    int K = 1000;  // kernel launches
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuModuleLoad(&module,
                            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                            "resource_limitation_kernel.ptx"));
    checkError(cuModuleGetFunction(&function, module, "_Z12dummy_kernelv"));

    for (int i = 0; i < K; i++) {
        int gridDimX = (i % M) + 1;
        int blockDimX = (i % N) + 1;
        void* args[] = {};
        checkError(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0,
                                  NULL, args, NULL));
        checkError(cuCtxSynchronize());
    }

    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));
}

TEST(CudaDriverApiTest, StreamCreation) {
    int N = 1000;
    CUdevice device;
    CUcontext context;
    CUstream stream[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    for (int i = 0; i < N; i++) {
        checkError(cuStreamCreate(
            &stream[i],
            CU_STREAM_DEFAULT));
        checkError(cuStreamDestroy(stream[i]));
    }

    checkError(cuCtxDestroy(context));
}

TEST(CudaDriverApiTest, EventCreation) {
    int N = 1000;
    CUdevice device;
    CUcontext context;
    CUevent event[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    for (int i = 0; i < N; i++) {
        checkError(cuEventCreate(
            &event[i],
            CU_EVENT_DEFAULT));
        checkError(cuEventDestroy(event[i]));
    }

    checkError(cuCtxDestroy(context));
}
