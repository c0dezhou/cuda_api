#include <cuda.h>
#include <iostream>

#define MAX_ITERATIONS 100

int main() {
    CUdevice dev1, dev2;
    CUcontext ctx1, ctx2;

    cuInit(0);

    cuDeviceGet(&dev1, 0);
    cuCtxCreate(&ctx1, 0, dev1);
    size_t free_mem_begin, total_mem, free_mem_end;

    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory

    cuDeviceGet(&dev2, 1);
    cuCtxCreate(&ctx2, 0, dev2);

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        cuCtxPushCurrent(ctx1);
        cuCtxPopCurrent(NULL);

        cuCtxPushCurrent(ctx2);
        cuCtxPopCurrent(NULL);
    }

    cuCtxSetCurrent(ctx1);
    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after context switching: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx2);
    cuCtxDestroy(ctx1);

    return 0;
}
