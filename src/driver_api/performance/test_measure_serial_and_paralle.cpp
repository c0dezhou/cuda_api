#include "test_utils.h"

#define SIZE 1024 * 1024  // 1 MiB
#define NUM_STREAMS 4

TEST(PERF, measure_serial_and_parallel) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&devPtr, SIZE * NUM_STREAMS);
    cuMemAllocHost(&hostPtr, SIZE * NUM_STREAMS);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuMemcpyHtoD(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "Sequential transfer took " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<CUstream> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamCreate(&streams[i], 0);
        cuMemcpyHtoDAsync(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE, streams[i]);
    }
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamSynchronize(streams[i]);
        cuStreamDestroy(streams[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Concurrent transfer took " << diff.count() << " ms" << std::endl;

    cuMemFreeHost(hostPtr);
    cuMemFree(devPtr);
    cuCtxDestroy(ctx);

}
