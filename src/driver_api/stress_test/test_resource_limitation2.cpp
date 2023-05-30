#include "test_utils.h"

TEST(CudaDriverApiTest, MemoryExhaustion_fragments) {
    int N = 1000 * 40;
    int M = 1024 * 1024; // 1MB
    CUdevice device;
    CUcontext context;
    CUdeviceptr devPtr[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    for (int i = 0; i < N; i++) {
        CUresult result =
            cuMemAlloc(&devPtr[i], M);
        if (result != CUDA_SUCCESS) {
            printf("cuMemAlloc failed: %d\n", result);
            break;
        }
        printf("Allocated %d MB of device memory\n", (i + 1));
    }

    for (int i = 0; i < N; i++) {
        if (devPtr[i]) {
            checkError(cuMemFree(devPtr[i]));
        }
    }

    checkError(cuCtxDestroy(context));
}

TEST(CudaDriverApiTest, MemoryExhaustion_largemem) {
    int N = 4;
    int M = 10 * 1024 * 1024 * 1024;  // 10G
    CUdevice device;
    CUcontext context;
    CUdeviceptr devPtr[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    for (int i = 0; i < N; i++) {
        CUresult result = cuMemAlloc(&devPtr[i], M);
        if (result != CUDA_SUCCESS) {
            printf("cuMemAlloc failed: %d\n", result);
            break;
        }
        printf("Allocated %d *10 GB of device memory\n", (i + 1));
    }

    for (int i = 0; i < N; i++) {
        if (devPtr[i]) {
            checkError(cuMemFree(devPtr[i]));
        }
    }

    checkError(cuCtxDestroy(context));
}


TEST(CudaDriverApiTest, ThreadTimeout) {
    GTEST_SKIP();
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;

    int N = 1024;
    int M = 1024;

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(
        cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC,
                    device));
    checkError(cuModuleLoad(&module,
                            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                            "resource_limitation_kernel.ptx"));
    checkError(cuModuleGetFunction(&function, module, "_Z15infinite_kernelv"));

    void* args[] = {};
    checkError(cuLaunchKernel(function, M, 1, 1, N, 1, 1, 0, NULL, args,
                              NULL));

    CUresult result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        printf("cuCtxSynchronize failed: %d\n", result);
        switch (result) {
            case CUDA_ERROR_LAUNCH_TIMEOUT:
                printf("The kernel exceeded the maximum execution time\n");
                break;
            case CUDA_ERROR_LAUNCH_FAILED:
                printf("The kernel launch failed for an unknown reason\n");
                break;
            default:
                printf("Some other error occurred\n");
                break;
        }
    }

    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));
}


TEST(CudaDriverApiTest, StreamEventSyncQuery) {
    int N = 1000;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUstream stream[N];
    CUevent event[N];

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/"
        "resource_limitation_kernel.ptx"));
    checkError(cuModuleGetFunction(&function, module, "_Z12dummy_kernelv"));

    for (int i = 0; i < N; i++) {
        checkError(cuStreamCreate(
            &stream[i],
            CU_STREAM_DEFAULT));
        checkError(cuEventCreate(
            &event[i],
            CU_EVENT_DEFAULT));
    }

    for (int i = 0; i < N; i++) {
        void* args[] = {};
        checkError(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream[i],
                                  args,
                                  NULL));
        checkError(cuEventRecord(event[i],
                                 stream[i]));
    }

    for (int i = 0; i < N; i++) {
        checkError(
            cuEventSynchronize(event[i]));
        CUresult result = cuEventQuery(event[i]);
        if (result == CUDA_SUCCESS) {
            printf("Event %d is completed.\n", i);
        } else if (result == CUDA_ERROR_NOT_READY) {
            printf("Event %d is not ready.\n", i);
        } else {
            printf("cuEventQuery failed: %d\n", result);
            break;
        }
    }

    for (int i = 0; i < N; i++) {
        checkError(cuStreamDestroy(stream[i]));
        checkError(cuEventDestroy(event[i]));
    }

    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));
}
