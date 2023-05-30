#include "loop_common.h"

CUresult memalloc_and_free(int non_use) {
    CUdeviceptr dptr;
    size_t size_d = 1024 * sizeof(float);
    checkError(cuMemAlloc(&dptr, size_d));
    checkError(cuMemFree(dptr));
    return CUDA_SUCCESS;
}

CUresult memallochost_and_freehost(int non_use) {
    float* hptr;
    size_t size_d = 10244 * sizeof(float);
    checkError(cuMemAllocHost((void**)&hptr, size_d));
    checkError(cuMemFreeHost(hptr));
    return CUDA_SUCCESS;
}

void checkmemResult(float* hptr, float* hptr_copy, int N) {
    for(int i =0; i< N; i++){
        ASSERT_EQ(hptr[i], hptr_copy[i]);
    }
}

void init_vec(float * vec, int N){
    for (int i = 0; i < N; i++) {
        vec[i] = i;
    }
}

void init_vec_zero(float* vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = 0;
    }
}

CUresult h2d_d2h(int non_use) {
    float* dptr;
    float hptr[10244] = {3.0f};
    float hprt_copy[10244] = {0.0f};
    size_t size_d = 10244 * sizeof(float);
    checkError(cuMemAlloc((CUdeviceptr*)&dptr, size_d));

    checkError(cuMemcpyHtoD((CUdeviceptr)dptr, hptr, size_d));
    checkError(cuMemcpyDtoH(hprt_copy, (CUdeviceptr)dptr, size_d));

    checkmemResult(hptr, hprt_copy, size_d / sizeof(float));
    checkError(cuMemFree((CUdeviceptr)dptr));
    return CUDA_SUCCESS;
}

CUresult h2d_d2hasync(CUstream stream) {
    float* dptr;
    float* hptr;
    float* hprt_copy;
    size_t size_d = 10244 * sizeof(float);
    checkError(cuMemAlloc((CUdeviceptr*)&dptr, size_d));
    checkError(cuMemAllocHost((void**)&hptr, size_d));
    checkError(cuMemAllocHost((void**)&hprt_copy, size_d));

    init_vec(hptr, 10244);

    checkError(
        cuMemcpyHtoDAsync((CUdeviceptr)dptr, hptr, size_d, stream));
    checkError(cuMemcpyDtoHAsync(hprt_copy, (CUdeviceptr)dptr, size_d, stream));
    cuStreamSynchronize(stream);
    checkmemResult(hptr, hprt_copy, size_d / sizeof(float));
    checkError(cuMemFree((CUdeviceptr)dptr));
    checkError(cuMemFreeHost(hptr));
    checkError(cuMemFreeHost(hprt_copy));
    
    return CUDA_SUCCESS;
}

TEST(LOOPSINGLE,Memory) {
    const int loop_times = 100;
    CUdevice device;
    CUcontext context;
    CUstream stream;
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuStreamCreate(&stream, 0));

    auto memalloc_and_freeFunc = makeFuncPtr(memalloc_and_free);
    auto memallochost_and_freehostFunc = makeFuncPtr(memallochost_and_freehost);
    auto h2d_d2hFunc = makeFuncPtr(h2d_d2h);
    auto h2d_d2hasyncFunc = makeFuncPtr(h2d_d2hasync);

    auto memalloc_and_freeParams = []() {
        return std::make_tuple(1);
    };
    auto memallochost_and_freehostParams = []() { return std::make_tuple(1); };
    auto h2d_d2hParams = []() { return std::make_tuple(1); };
    auto h2d_d2hasyncParams = [&stream]() { return std::make_tuple(stream); };


    PRINT_FUNCNAME(loopFuncPtr(loop_times, memalloc_and_freeFunc,
                               memalloc_and_freeParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, memallochost_and_freehostFunc,
                               memallochost_and_freehostParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, h2d_d2hFunc, h2d_d2hParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, h2d_d2hasyncFunc, h2d_d2hasyncParams()));

    cuStreamDestroy(stream);
    cuCtxDestroy(context);
}