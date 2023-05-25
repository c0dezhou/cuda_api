#include "test_utils.h"

#define SIZE 1024

TEST(PERF, overall_mem_usage) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem);

    cuMemAlloc(&devPtr, SIZE);

    cuMemGetInfo(&free_mem_end, &total_mem);

    double mem_used_mb = (double)(free_mem_begin - free_mem_end) / (1024.0 * 1024.0);

    std::cout << "Memory used by CUDA operations: " << mem_used_mb << " MB" << std::endl;
    std::cout << "Memory usage: "
              << SIZE / mem_used_mb * 1024.0 * 1024.0 <<
        std::endl;

    cuMemFree(devPtr);
    cuCtxDestroy(ctx);

}
