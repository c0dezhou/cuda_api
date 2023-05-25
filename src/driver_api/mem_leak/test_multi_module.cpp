#include "test_utils.h"

#define MAX_MODULES 1000

TEST(MEMLEAK, multi_modules) {
    CUdevice dev;
    CUcontext ctx;
    CUmodule cuModule[MAX_MODULES];
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem);

    for (int i = 0; i < MAX_MODULES; i++) {
        cuModuleLoad(
            &cuModule[i],
            "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    }

    cuMemGetInfo(&free_mem_end, &total_mem);
    std::cout << "Memory usage after loading " << MAX_MODULES << " modules: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    for (int i = 0; i < MAX_MODULES; i++) {
        cuModuleUnload(cuModule[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem);
    std::cout << "Memory usage after unloading modules: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
}
