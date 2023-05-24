#include <cuda.h>
#include <iostream>

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr d_A;
    size_t size = 1000;
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory
    cuCtxCreate(&ctx, 0, dev);
    cuMemAlloc(&d_A, size * sizeof(float));

    // cuMemFree(d_A); // Uncomment this to free memory
    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory

    std::cout << "Memory usage: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
    return 0;
}
