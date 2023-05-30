#include "test_utils.h"

#define kNumIterations 10
#define kDataSize 1024 * 1024 * 1024

TEST(PERF, d2hbw) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, 0));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    CUdeviceptr d_data;
    checkError(cuMemAlloc(&d_data, kDataSize));

    uint8_t* h_data_pinned;
    checkError(cuMemAllocHost((void**)&h_data_pinned, kDataSize));

    uint8_t* h_data_non_pinned = new uint8_t[kDataSize];

    CUevent startEvent, stopEvent;
    checkError(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));

    // Warmup
    checkError(cuMemcpyDtoH(h_data_pinned, d_data, kDataSize));
    checkError(cuMemcpyDtoH(h_data_non_pinned, d_data, kDataSize));

    double totalElapsedTimePinned = 0.0;
    double totalElapsedTimeNonPinned = 0.0;
    for (int i = 0; i < kNumIterations; i++) {
        // Pinned memory
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyDtoH(h_data_pinned, d_data, kDataSize));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimePinned;
        checkError(
            cuEventElapsedTime(&elapsedTimePinned, startEvent, stopEvent));
        totalElapsedTimePinned += elapsedTimePinned;

        // Non-pinned memory
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyDtoH(h_data_non_pinned, d_data, kDataSize));
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

    std::cout << "Pinned D2H Bandwidth: " << pinnedBandwidth << " GB/s"
              << std::endl;
    std::cout << "Non-Pinned D2H Bandwidth: " << nonPinnedBandwidth << " GB/s"
              << std::endl;

    // Cleanup
    checkError(cuMemFree(d_data));
    checkError(cuMemFreeHost(h_data_pinned));
    delete[] h_data_non_pinned;
    checkError(cuEventDestroy(startEvent));
    checkError(cuEventDestroy(stopEvent));
    checkError(cuCtxDestroy(cuContext));
}
