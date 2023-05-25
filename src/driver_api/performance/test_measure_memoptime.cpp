#include "test_utils.h"

typedef void (*transferTestFunc)(CUdeviceptr, void*, size_t);

void testHtoD(CUdeviceptr devPtr, void* hostPtr, size_t size) {
    cuMemcpyHtoD(devPtr, hostPtr, size);
}

void testDtoH(CUdeviceptr devPtr, void* hostPtr, size_t size) {
    cuMemcpyDtoH(hostPtr, devPtr, size);
}


void runTest(transferTestFunc test, const char* testName, CUdeviceptr devPtr, void* hostPtr, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();

    test(devPtr, hostPtr, size);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff = end - start;

    std::cout << testName << " transferred " << size << " bytes in "
              << diff.count() << " ms" << std::endl;
}

TEST(PERF, measure_memoption) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;
    size_t size = 1024 * 1024;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&devPtr, size);
    cuMemAllocHost(&hostPtr, size);

    std::vector<std::pair<transferTestFunc, const char*>> tests = {
        {testHtoD, "Host to Device"},
        {testDtoH, "Device to Host"}
    };

    for (auto& test : tests) {
        runTest(test.first, test.second, devPtr, hostPtr, size);
    }

    cuMemFreeHost(hostPtr);
    cuMemFree(devPtr);

    cuCtxDestroy(ctx);

}
