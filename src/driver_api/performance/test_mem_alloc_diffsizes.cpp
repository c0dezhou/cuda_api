#include <cuda.h>
#include <iostream>
#include <chrono>

#define MAX_POW 25

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    size_t size;

    // Initialize CUDA
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    for (int i = 0; i <= MAX_POW; i++) {
        size = 1 << i; // size = 2^i bytes

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // Allocate and deallocate device memory
        cuMemAlloc(&devPtr, size);
        cuMemFree(devPtr);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute the difference between the two times in milliseconds
        std::chrono::duration<double, std::milli> diff = end - start;

        std::cout << "Allocated and deallocated " << size << " bytes in "
                  << diff.count() << " ms" << std::endl;
    }

    // Destroy CUDA context
    cuCtxDestroy(ctx);

    return 0;
}
