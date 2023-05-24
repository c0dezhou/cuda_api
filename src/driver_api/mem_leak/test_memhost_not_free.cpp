#include <cuda.h>
#include <iostream>

#define MAX_ALLOCATIONS 100

int main() {
    CUdevice dev;
    CUcontext ctx;
    void* hostMem[MAX_ALLOCATIONS];
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory

    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        cuMemAllocHost(&hostMem[i], 1024);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after allocating " << MAX_ALLOCATIONS << " host memory blocks: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    // Free host memory
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        cuMemFreeHost(hostMem[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after freeing host memory blocks: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
    return 0;
}
