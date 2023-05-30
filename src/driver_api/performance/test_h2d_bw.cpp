#include "test_utils.h"

#define kNumIterations 10
#define kDataSize 1024 * 1024 * 1024  // 1 GB

// 都没写async的 后面可以补
TEST(PERF, h2dbw) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, 0));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    uint8_t* h_data_pinned;
    checkError(cuMemAllocHost((void**)&h_data_pinned, kDataSize));

    uint8_t* h_data_non_pinned = new uint8_t[kDataSize];

    CUdeviceptr d_data;
    checkError(cuMemAlloc(&d_data, kDataSize));

    CUevent startEvent, stopEvent;
    checkError(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));

    // warmup
    checkError(cuMemcpyHtoD(d_data, h_data_pinned, kDataSize));
    checkError(cuMemcpyHtoD(d_data, h_data_non_pinned, kDataSize));

    double totalElapsedTimePinned = 0.0;
    double totalElapsedTimeNonPinned = 0.0;
    for (int i = 0; i < kNumIterations; i++) {
        // pinned memory
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyHtoD(d_data, h_data_pinned, kDataSize));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimePinned;
        checkError(
            cuEventElapsedTime(&elapsedTimePinned, startEvent, stopEvent));
        totalElapsedTimePinned += elapsedTimePinned;

        // Non-pinned memory
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyHtoD(d_data, h_data_non_pinned, kDataSize));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimeNonPinned;
        checkError(
            cuEventElapsedTime(&elapsedTimeNonPinned, startEvent, stopEvent));
        totalElapsedTimeNonPinned += elapsedTimeNonPinned;
    }

    double averageElapsedTimePinned = totalElapsedTimePinned / kNumIterations;
    double averageElapsedTimeNonPinned =
        totalElapsedTimeNonPinned / kNumIterations;

    double pinnedBandwidth =
        (kDataSize * 1e-6) / (averageElapsedTimePinned * 1e-3);
    double nonPinnedBandwidth =
        (kDataSize * 1e-6) / (averageElapsedTimeNonPinned * 1e-3);

    std::cout << "Pinned H2D Bandwidth: " << pinnedBandwidth << " GB/s"
              << std::endl;
    std::cout << "Non-Pinned H2D Bandwidth: " << nonPinnedBandwidth << " GB/s"
              << std::endl;

    checkError(cuMemFreeHost(h_data_pinned));
    delete[] h_data_non_pinned;
    checkError(cuMemFree(d_data));
    checkError(cuEventDestroy(startEvent));
    checkError(cuEventDestroy(stopEvent));
    checkError(cuCtxDestroy(cuContext));
}
