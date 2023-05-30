#include "test_utils.h"

#define kNumIterations 10
#define kDataSize 1024 * 1024 * 1024  // 1 GB

TEST(PERF, d2dbw) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, 0));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    // CUstream stream;
    // checkError(cuStreamCreate(&stream, 0));

    CUdeviceptr d_data_src, d_data_dst;
    checkError(cuMemAlloc(&d_data_src, kDataSize));
    checkError(cuMemAlloc(&d_data_dst, kDataSize));

    CUdeviceptr d_data_async_src, d_data_async_dst;
    checkError(cuMemAlloc(&d_data_async_src, kDataSize));
    checkError(cuMemAlloc(&d_data_async_dst, kDataSize));

    CUevent startEvent, stopEvent;
    checkError(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));

    // Warmup
    checkError(cuMemcpy(d_data_dst, d_data_src, kDataSize));
    checkError(cuMemcpyAsync(d_data_async_dst, d_data_async_src, kDataSize, 0));

    double totalElapsedTimeSync = 0.0;
    double totalElapsedTimeAsync = 0.0;
    for (int i = 0; i < kNumIterations; i++) {
        // sync
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpy(d_data_dst, d_data_src, kDataSize));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimeSync;
        checkError(
            cuEventElapsedTime(&elapsedTimeSync, startEvent, stopEvent));
        totalElapsedTimeSync += elapsedTimeSync;

        // async
        checkError(cuEventRecord(startEvent, 0));
        checkError(
            cuMemcpyAsync(d_data_async_dst, d_data_async_src, kDataSize, 0));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimeAsync;
        checkError(
            cuEventElapsedTime(&elapsedTimeAsync, startEvent, stopEvent));
        totalElapsedTimeAsync += elapsedTimeAsync;
    }

    double averageElapsedTimeSync = totalElapsedTimeSync / kNumIterations;
    double averageElapsedTimeAsync =
        totalElapsedTimeAsync / kNumIterations;

    double pinnedBandwidth =
        (kDataSize * 1e-6) / (averageElapsedTimeSync * 1e-3);
    double nonSyncBandwidth =
        (kDataSize * 1e-6) / (averageElapsedTimeAsync * 1e-3);

    std::cout << "Sync D2D Bandwidth: " << pinnedBandwidth << " GB/s"
              << std::endl;
    std::cout << "Async D2D Bandwidth: " << nonSyncBandwidth << " GB/s"
              << std::endl;

    checkError(cuMemFree(d_data_src));
    checkError(cuMemFree(d_data_dst));
    checkError(cuMemFree(d_data_async_src));
    checkError(cuMemFree(d_data_async_dst));
    checkError(cuEventDestroy(startEvent));
    checkError(cuEventDestroy(stopEvent));
    checkError(cuCtxDestroy(cuContext));
}
