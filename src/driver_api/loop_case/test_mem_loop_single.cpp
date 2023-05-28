#include "loop_common.h"

CUresult memalloc_and_free(int flag) {
    CUdeviceptr dptr;
    size_t size_d = 1024 * sizeof(float);
    checkError(cuMemAlloc(&dptr, size_d));
    checkError(cuMemFree(dptr));
    return CUDA_SUCCESS;
}

TEST(LOOPSINGLE,Memory) {
    const int loop_times = 500;
    CUdevice device;
    CUcontext context;
    LOOP(cuInit(0), 1);
    LOOP(cuDeviceGet(&device, 0), 1);
    LOOP(cuCtxCreate(&context, 0, device), 1);

    auto cuMemAllocFunc = makeFuncPtr(cuMemAlloc);
    auto cuMemAllocHostFunc = makeFuncPtr(cuMemAllocHost);
    auto cuMemcpyDtoHFunc = makeFuncPtr(cuMemcpyDtoH);
    auto cuMemcpyDtoHAsyncFunc = makeFuncPtr(cuMemcpyDtoHAsync);
    auto cuMemcpyHtoDFunc = makeFuncPtr(cuMemcpyHtoD);
    auto cuCtxGetCurrentFunc = makeFuncPtr(cuMemcpyHtoDAsync);

    auto memalloc_and_freeFunc = makeFuncPtr(memalloc_and_free);
    auto memalloc_and_freeParams = []() {
        return std::make_tuple(1);
    };

    auto cuMemAllocParams = [&device]() {
        float* dptr;
        size_t size_d = 1024 * sizeof(float);
        return std::make_tuple((CUdeviceptr*)dptr, size_d);
    };

    PRINT_FUNCNAME(loopFuncPtr(loop_times, memalloc_and_freeFunc,
                               memalloc_and_freeParams()));

    cuCtxDestroy(context);

}