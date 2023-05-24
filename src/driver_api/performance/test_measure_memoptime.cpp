#include <cuda.h>
#include <iostream>
#include <chrono>
#include <vector>

// Create a function type for our tests
typedef void (*transferTestFunc)(CUdeviceptr, void*, size_t);

// Test scenarios
void testHtoD(CUdeviceptr devPtr, void* hostPtr, size_t size) {
    cuMemcpyHtoD(devPtr, hostPtr, size);
}

void testDtoH(CUdeviceptr devPtr, void* hostPtr, size_t size) {
    cuMemcpyDtoH(hostPtr, devPtr, size);
}

// Add more test scenarios here...

// Test runner
void runTest(transferTestFunc test, const char* testName, CUdeviceptr devPtr, void* hostPtr, size_t size) {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Run the test
    test(devPtr, hostPtr, size);

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute the difference between the two times in milliseconds
    std::chrono::duration<double, std::milli> diff = end - start;

    std::cout << testName << " transferred " << size << " bytes in "
              << diff.count() << " ms" << std::endl;
}

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;
    size_t size = 1024 * 1024;  // 1 MiB

    // Initialize CUDA
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    // Allocate host and device memory
    cuMemAlloc(&devPtr, size);
    cuMemAllocHost(&hostPtr, size);

    // Define the tests
    std::vector<std::pair<transferTestFunc, const char*>> tests = {
        {testHtoD, "Host to Device"},
        {testDtoH, "Device to Host"}
        // Add more tests here...
    };

    // Run the tests
    for (auto& test : tests) {
        runTest(test.first, test.second, devPtr, hostPtr, size);
    }

    // Free host and device memory
    cuMemFreeHost(hostPtr);
    cuMemFree(devPtr);

    // Destroy CUDA context
    cuCtxDestroy(ctx);

    return 0;
}
