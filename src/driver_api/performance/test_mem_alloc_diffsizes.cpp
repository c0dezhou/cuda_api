#include "test_utils.h"

#define MAX_POW 25

TEST(PERF, alloc_diff_size) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    size_t size;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    for (int i = 0; i <= MAX_POW; i++) {
        size = 1 << i;

        auto start = std::chrono::high_resolution_clock::now();

        cuMemAlloc(&devPtr, size);
        cuMemFree(devPtr);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> diff = end - start;

        std::cout << "Allocated and deallocated " << size << " bytes in "
                  << diff.count() << " ms" << std::endl;
    }

    cuCtxDestroy(ctx);
}
