#include <cuda.h>
#include <iostream>

#define SIZE 1024

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    size_t free_mem_begin, total_mem, free_mem_end;

    // Initialize CUDA
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    // Get the amount of free memory on the device before allocation
    cuMemGetInfo(&free_mem_begin, &total_mem);

    // Allocate device memory
    cuMemAlloc(&devPtr, SIZE);

    // Get the amount of free memory on the device after allocation
    cuMemGetInfo(&free_mem_end, &total_mem);

    // Compute the memory used by our CUDA operations (in MB)
    double mem_used_mb = (double)(free_mem_begin - free_mem_end) / (1024.0 * 1024.0);

    std::cout << "Memory used by CUDA operations: " << mem_used_mb << " MB" << std::endl;

    // Free device memory and destroy CUDA context
    cuMemFree(devPtr);
    cuCtxDestroy(ctx);

    return 0;
}
