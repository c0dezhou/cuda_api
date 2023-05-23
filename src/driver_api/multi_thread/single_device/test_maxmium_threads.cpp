#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath> // For std::abs()

// CUDA kernel
__global__ void kernel(float* data, int startIndex, int endIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= startIndex && i < endIndex) {
        data[i] *= 2.0f;
    }
}

// CUDA error checking macro
#define CHECK_CUDA(call)                                                             \
    {                                                                                 \
        CUresult result = call;                                                       \
        if (result != CUDA_SUCCESS) {                                                 \
            const char* errorStr;                                                     \
            cuGetErrorString(result, &errorStr);                                       \
            std::cerr << "CUDA error: " << errorStr << std::endl;                      \
            std::exit(1);                                                             \
        }                                                                             \
    }

void performCUDAOperations(int threadId, CUcontext context, CUdeviceptr d_data, int size) {
    // Calculate the portion of data for this thread
    int numThreads = std::thread::hardware_concurrency();
    int dataSizePerThread = size / numThreads;
    int startIndex = threadId * dataSizePerThread;
    int endIndex = startIndex + dataSizePerThread;
    if (threadId == numThreads - 1) {
        // Last thread takes care of the remaining elements if size % numThreads != 0
        endIndex += size % numThreads;
    }

    // Set the CUDA context for the current thread
    CHECK_CUDA(cuCtxSetCurrent(context));

    // Create CUDA stream for the current thread
    CUstream stream;
    CHECK_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // Perform CUDA operations
    CUfunction function;
    CHECK_CUDA(cuModuleGetFunction(&function, module, "kernel"));

    void* args[] = {&d_data, &startIndex, &endIndex};
    CHECK_CUDA(cuLaunchKernel(function, gridSize.x, 1, 1, blockSize.x, 1, 1, 0, stream, args, nullptr));

    // Synchronize the CUDA stream
    CHECK_CUDA(cuStreamSynchronize(stream));

    // Destroy the CUDA stream
    CHECK_CUDA(cuStreamDestroy(stream));
}

bool verifyResult(const std::vector<float>& data, float expectedValue) {
    for (const auto& element : data) {
        if (std::abs(element - expectedValue) > 1e-5) {
            return false; // Result doesn't match expected value
        }
    }
    return true; // Result matches expected value
}

int main() {
    const int dataSize = 10000;
    const int numThreads = std::thread::hardware_concurrency();

    // Initialize CUDA driver API
    CHECK_CUDA(cuInit(0));

    // Get the CUDA device
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    // Create CUDA context
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Allocate memory on the device
    CUdeviceptr d_data;
    CHECK_CUDA(cuMemAlloc(&d_data, dataSize * sizeof(float)));

    // Perform CUDA operations in multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back(performCUDAOperations, i, context, d_data, dataSize);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Copy the data back to the host
    std::vector<float> h_data(dataSize);
    CHECK_CUDA(cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float)));

    // Verify the result
    bool result = verifyResult(h_data, 0.0f);
    std::cout << "Result: " << (result ? "Passed" : "Failed") << std::endl;

    // Free the device memory
    CHECK_CUDA(cuMemFree(d_data));

    // Destroy the CUDA context
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}
