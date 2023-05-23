#include <iostream>
#include <vector>
#include <thread>
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUDA kernel function 1
__global__ void kernel1(float* data, int startIndex, int endIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= startIndex && tid < endIndex) {
        // Perform computation on data[tid] for kernel 1
        data[tid] *= 2.0f;
    }
}

// CUDA kernel function 2
__global__ void kernel2(float* data, int startIndex, int endIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= startIndex && tid < endIndex) {
        // Perform computation on data[tid] for kernel 2
        data[tid] += 3.0f;
    }
}

bool verifyResult(const std::vector<float>& data, float expectedValue, int startIndex, int endIndex) {
    for (int i = startIndex; i < endIndex; i++) {
        if (data[i] != expectedValue) {
            return false;
        }
    }
    return true;
}


// CUDA thread function
void cudaThread(int threadId, float* d_data, int dataSize, int startIndex, int endIndex) {
    // Create CUDA stream
    CUstream stream;
    cuStreamCreate(&stream, 0);

    // Calculate the grid and block dimensions
    int blockSize = 256;
    int gridSize = (endIndex - startIndex + blockSize - 1) / blockSize;

    // Launch the kernel based on the threadId
    if (threadId % 2 == 0) {
        // Even threadId executes kernel1
        CUfunction cuKernel;
        cuModuleGetFunction(&cuKernel, cuModule, "kernel1");
        void* kernelArgs[] = { &d_data, &startIndex, &endIndex };
        cuLaunchKernel(cuKernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream, kernelArgs, nullptr);
    } else {
        // Odd threadId executes kernel2
        CUfunction cuKernel;
        cuModuleGetFunction(&cuKernel, cuModule, "kernel2");
        void* kernelArgs[] = { &d_data, &startIndex, &endIndex };
        cuLaunchKernel(cuKernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream, kernelArgs, nullptr);
    }

    // Synchronize the stream
    cuStreamSynchronize(stream);

    // Destroy the stream
    cuStreamDestroy(stream);
}

int main() {
    int dataSize = 1000000; // Size of the data
    int numThreads = 16; // Number of threads
    int dataSizePerThread = dataSize / numThreads; // Data size per thread

    // Allocate host memory for data
    std::vector<float> h_data(dataSize, 1.0f);

    // Allocate device memory for data
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, dataSize * sizeof(float));

    // Copy data from host to device
    cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float));

    // Create threads
    std::vector<std::thread> threads;
    for (int threadId = 0; threadId < numThreads; ++threadId) {
        int startIndex = threadId * dataSizePerThread;
        int endIndex = startIndex + dataSizePerThread;
        if (threadId == numThreads - 1) {
            endIndex += dataSize % numThreads;
        }
        threads.emplace_back(cudaThread, threadId, d_data, dataSize, startIndex, endIndex);
    }

    // Wait for all threads to finish
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Copy data from device to host
    cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float));

    bool result = verifyResult(h_data, 2.0f, 0, dataSize/2); // Verify kernel1 result
    std::cout << "Kernel 1 result verification: " << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(h_data, 4.0f, dataSize/2, dataSize); // Verify kernel2 result
    std::cout << "Kernel 2 result verification: " << (result ? "Passed" : "Failed") << std::endl;


    // Free device memory
    cuMemFree(d_data);

    return 0;
}
