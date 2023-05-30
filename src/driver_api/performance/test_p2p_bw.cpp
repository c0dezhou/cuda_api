#include "test_utils.h"

#define kNumIterations 10
#define kDataSize 1024 * 1024 * 1024  // 1 GB

TEST(PERF, p2pbw) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, 0));

    CUcontext cuContext, peerContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    CUdevice peerDevice;
    checkError(cuDeviceGet(&peerDevice, 1));
    checkError(cuCtxCreate(&peerContext, 0, peerDevice));

    checkError(cuCtxSetCurrent(cuContext));
    CUdeviceptr d_data_src, d_data_async_src;
    checkError(cuMemAlloc(&d_data_src, kDataSize));
    checkError(cuMemAlloc(&d_data_async_src, kDataSize));

    checkError(cuCtxSetCurrent(peerContext));
    CUdeviceptr d_data_dst, d_data_async_dst;
    checkError(cuMemAlloc(&d_data_dst, kDataSize));
    checkError(cuMemAlloc(&d_data_async_dst, kDataSize));

    checkError(cuCtxSetCurrent(cuContext));
    CUevent startEvent, stopEvent;
    checkError(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));

    checkError(cuCtxEnablePeerAccess(cuContext, 0));
    checkError(cuCtxEnablePeerAccess(peerContext, 0));

    // Warmup
    checkError(
        cuMemcpyPeer(d_data_dst, 0, d_data_src, 0, kDataSize));
    checkError(cuMemcpyPeer(d_data_async_dst, 0, d_data_async_src, 0,
                            kDataSize));

    double totalElapsedTimeSync = 0.0;
    double totalElapsedTimeAsync = 0.0;
    for (int i = 0; i < kNumIterations; i++) {
        // Sync memory transfer
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyPeer(d_data_dst, cuContext, d_data_src, peerContext,
                                kDataSize));
        checkError(cuEventRecord(stopEvent, 0));
        checkError(cuEventSynchronize(stopEvent));
        float elapsedTimeSync;
        checkError(
            cuEventElapsedTime(&elapsedTimeSync, startEvent, stopEvent));
        totalElapsedTimeSync += elapsedTimeSync;

        // async memory transfer
        checkError(cuEventRecord(startEvent, 0));
        checkError(cuMemcpyPeerAsync(d_data_async_dst, cuContext,
                                     d_data_async_src, peerContext, kDataSize, 0));
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

    std::cout << "Sync P2P Bandwidth: " << pinnedBandwidth << " GB/s"
              << std::endl;
    std::cout << "Non-Sync P2P Bandwidth: " << nonSyncBandwidth << " GB/s"
              << std::endl;

    // Cleanup
    checkError(cuMemFree(d_data_src));
    checkError(cuMemFree(d_data_dst));
    checkError(cuMemFree(d_data_async_src));
    checkError(cuMemFree(d_data_async_dst));
    checkError(cuEventDestroy(startEvent));
    checkError(cuEventDestroy(stopEvent));
    checkError(cuCtxDisablePeerAccess(cuContext));
    checkError(cuCtxDisablePeerAccess(peerContext));
    checkError(cuCtxDestroy(cuContext));
    checkError(cuCtxDestroy(peerContext));

}
