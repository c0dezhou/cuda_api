#include <cuda.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

#define SIZE 1024 * 1024  // 1 MiB
#define NUM_STREAMS 4

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&devPtr, SIZE * NUM_STREAMS);
    cuMemAllocHost(&hostPtr, SIZE * NUM_STREAMS);

    // Sequential transfer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuMemcpyHtoD(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "Sequential transfer took " << diff.count() << " ms" << std::endl;

    // Concurrent transfer using streams
    start = std::chrono::high_resolution_clock::now();
    std::vector<CUstream> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamCreate(&streams[i], 0);
        cuMemcpyHtoDAsync(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE, streams[i]);
    }
    // Wait for all streams to finish
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

    return 0;
}
