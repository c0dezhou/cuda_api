#include "test_utils.h"

// #define SIZE 1024 * 1024  // 1 MiB
#define NUM_STREAMS 4

TEST(COMBINE, n_stream_n_event) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;
    CUmodule cuModule = 0;
    CUfunction cuFunction = 0;
    int SIZE = 1024 * 1024;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&devPtr, SIZE * NUM_STREAMS);
    cuMemAllocHost(&hostPtr, SIZE * NUM_STREAMS);

    // Load CUDA module
    cuModuleLoad(&cuModule,
                 "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                 "cuda_kernel.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "_Z14vec_multiply_2Pfi");

    std::vector<CUstream> streams(NUM_STREAMS);
    std::vector<CUevent> startEvents(NUM_STREAMS);
    std::vector<CUevent> stopEvents(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cuStreamCreate(&streams[i], 0);
        cuEventCreate(&startEvents[i], 0);
        cuEventCreate(&stopEvents[i], 0);

        cuMemcpyHtoDAsync(devPtr + i * SIZE, (char*)hostPtr + i * SIZE, SIZE,
                          streams[i]);
        void* args[] = {&devPtr, &SIZE};
        cuLaunchKernel(cuFunction, SIZE / 256, 1, 1, 256, 1, 1, 0, streams[i],
                       args, NULL);
        cuMemcpyDtoHAsync((char*)hostPtr + i * SIZE, devPtr + i * SIZE, SIZE,
                          streams[i]);

        cuEventRecord(startEvents[i], streams[i]);
        cuEventRecord(stopEvents[i], streams[i]);
    }

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

}
