#include "test_utils.h"

#define MAX_EVENTS 10000

TEST(MEMLEAK, multi_event) {
    CUdevice dev;
    CUcontext ctx;
    CUevent cuEvent[MAX_EVENTS];
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem);

    for (int i = 0; i < MAX_EVENTS; i++) {
        cuEventCreate(&cuEvent[i], 0);
    }

    cuMemGetInfo(&free_mem_end, &total_mem);
    std::cout << "Memory usage after creating " << MAX_EVENTS << " events: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    for (int i = 0; i < MAX_EVENTS; i++) {
        cuEventDestroy(cuEvent[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem);
    std::cout << "Memory usage after destroying events: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
}
