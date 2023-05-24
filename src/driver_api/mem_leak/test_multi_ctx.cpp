#include <cuda.h>
#include <iostream>

#define MAX_CONTEXTS 10
#define SIZE_PER_CONTEXT (50 * 1024 * 1024) // 50 MB per context

int main() {
    CUdevice dev;
    CUcontext ctx[MAX_CONTEXTS];
    CUdeviceptr d_A[MAX_CONTEXTS];
    size_t size = SIZE_PER_CONTEXT;
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory

    for (int i = 0; i < MAX_CONTEXTS; i++) {
        cuCtxCreate(&ctx[i], 0, dev);
        cuMemAlloc(&d_A[i], size);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after allocating memory in " << MAX_CONTEXTS << " contexts: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    // Attempt to allocate more memory than is available
    CUcontext ctx_extra;
    CUdeviceptr d_A_extra;
    cuCtxCreate(&ctx_extra, 0, dev);
    CUresult result = cuMemAlloc(&d_A_extra, size);
    if (result != CUDA_SUCCESS) {
        std::cout << "Failed to allocate extra memory: not enough memory available\n";
    }

    // Free memory and destroy contexts
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        cuMemFree(d_A[i]);
        cuCtxDestroy(ctx[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after freeing memory: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    return 0;
}
