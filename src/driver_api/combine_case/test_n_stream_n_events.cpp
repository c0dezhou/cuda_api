#include <cuda.h>
#include <iostream>
#include <chrono>
#include <vector>

#define SIZE 1024 * 1024  // 1 MiB
#define NUM_STREAMS 4

// Simple kernel to double the input values
__global__ void doubleValues(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;
    CUmodule cuModule = 0;
    CUfunction cuFunction = 0;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&devPtr, SIZE * NUM_STREAMS);
    cuMemAllocHost(&hostPtr, SIZE * NUM_STREAMS);

    // Load CUDA module
    cuModuleLoad(&cuModule, "doubleValues.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "doubleValues");

    std::vector<CUstream> streams(NUM_STREAMS);
    std::vector<CUevent> startEvents(NUM_STREAMS);
    std::vector<CUevent> stopEvents(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamCreate(&streams[i], 0);
        cuEventCreate(&startEvents[i], 0);
        cuEventCreate(&stopEvents[i], 0);

        cuMemcpyHtoDAsync(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE, streams[i]);
        void* args[] = { &devPtr, &SIZE };
        cuLaunchKernel(cuFunction, SIZE / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL);
        cuMemcpyDtoHAsync((char*)hostPtr + i * SIZE, devPtr + i * SIZE, SIZE, streams[i]);

        // Record start and stop events
        cuEventRecord(startEvents[i], streams[i]);
        cuEventRecord(stopEvents[i], streams[i]);
    }

    // Wait for all streams to finish and calculate time for each stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamSynchronize(streams[i]);

        float time;
        cuEventElapsedTime(&time, startEvents[i], stopEvents[i]);
        std::cout << "Stream " << i << " took " << time << " ms" << std::endl;

        cuStreamDestroy(streams[i]);
        cuEventDestroy(startEvents[i]);
        cuEventDestroy(stopEvents[i]);
    }

    cuMemFreeHost(hostPtr);
    cuMemFree(devPtr);
    cuCtxDestroy(ctx);

    return 0;
}
