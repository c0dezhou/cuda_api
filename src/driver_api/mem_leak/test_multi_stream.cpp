#include <cuda.h>
#include <iostream>

#define MAX_STREAMS 100

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUstream cuStream[MAX_STREAMS];
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory

    for (int i = 0; i < MAX_STREAMS; i++) {
        cuStreamCreate(&cuStream[i], 0);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after creating " << MAX_STREAMS << " streams: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    // Destroy streams
    for (int i = 0; i < MAX_STREAMS; i++) {
        cuStreamDestroy(cuStream[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after destroying streams: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
    return 0;
}
