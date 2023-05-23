#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath> // For std::abs()

// CUDA kernel 1
__global__ void kernel1(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] *= 2.0f;
    }
}

// CUDA kernel 2
__global__ void kernel2(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] += 3.0f;
    }
}

// CUDA kernel 3
__global__ void kernel3(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] -= 1.0f;
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
    // Create CUDA events
    CUevent event1, event2;

    CHECK_CUDA(cuEventCreate(&event1, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&event2, CU_EVENT_DEFAULT));

    // Calculate the portion of data for this thread
    int dataSizePerThread = size / 4;
    int startIndex = threadId * dataSizePerThread;
    int endIndex = startIndex + dataSizePerThread;

    // Select kernel based on thread ID
    CUfunction kernel;
    if (threadId == 0)
        kernel = kernel1;
    else if (threadId == 1)
        kernel = kernel2;
    else
        kernel = kernel3;

    // Perform CUDA operations
    CHECK_CUDA(cuEventRecord(event1, 0));
    CHECK_CUDA(cuLaunchKernel(kernel, (size + 255) / 256, 1, 1, 256, 1, 1, 0, 0, &d_data, &size));
    CHECK_CUDA(cuEventRecord(event2, 0));

    // Synchronize events
    CHECK_CUDA(cuEventSynchronize(event2));

    // Print results
    std::cout << "Thread " << threadId << " finished" << std::endl;

    // Cleanup
    CHECK_CUDA(cuEventDestroy(event1));
    CHECK_CUDA(cuEventDestroy(event2));
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
    const int numThreads = 4;

    // Initialize CUDA
    CHECK_CUDA(cuInit(0));

    // Create CUDA context
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, 0));

    // Load CUDA module
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "path/to/cuda_module.ptx"));

    // Get CUDA kernel functions
    CUfunction kernel1, kernel2, kernel3;
    CHECK_CUDA(cuModuleGetFunction(&kernel1, module, "kernel1"));
    CHECK_CUDA(cuModuleGetFunction(&kernel2, module, "kernel2"));
    CHECK_CUDA(cuModuleGetFunction(&kernel3, module, "kernel3"));

    // Allocate device memory
    CUdeviceptr d_data;
    CHECK_CUDA(cuMemAlloc(&d_data, dataSize * sizeof(float)));

    // Initialize host data
    std::vector<float> h_data(dataSize, 1.0f);

    // Copy host data to device
    CHECK_CUDA(cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float)));

    // Create threads and perform CUDA operations
    std::vector<std::thread> threads(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(performCUDAOperations, i, context, d_data, dataSize);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Copy device data back to host
    CHECK_CUDA(cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float)));

    // Verify the result
    bool resultIsValid = verifyResult(h_data, 4.0f); // Expected value based on the performed operations

    // Print result validity
    std::cout << "Result is " << (resultIsValid ? "valid" : "invalid") << std::endl;

    // Cleanup
    CHECK_CUDA(cuMemFree(d_data));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}
