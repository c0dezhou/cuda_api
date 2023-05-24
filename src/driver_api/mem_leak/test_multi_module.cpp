#include <cuda.h>
#include <iostream>

#define MAX_MODULES 100

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUmodule cuModule[MAX_MODULES];
    size_t free_mem_begin, total_mem, free_mem_end;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemGetInfo(&free_mem_begin, &total_mem); // get initial free memory

    for (int i = 0; i < MAX_MODULES; i++) {
        cuModuleLoad(&cuModule[i], "cuda_module.ptx");
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after loading " << MAX_MODULES << " modules: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    // Unload modules
    for (int i = 0; i < MAX_MODULES; i++) {
        cuModuleUnload(cuModule[i]);
    }

    cuMemGetInfo(&free_mem_end, &total_mem); // get final free memory
    std::cout << "Memory usage after unloading modules: " << (free_mem_begin - free_mem_end) / 1024.0 / 1024.0 << " MB\n";

    cuCtxDestroy(ctx);
    return 0;
}
