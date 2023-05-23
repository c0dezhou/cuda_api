// Hybrid CPU-GPU Computing: In hybrid computing environments, where both CPU and GPU resources are available, multiple threads can be used to leverage both CPU and GPU processing power. Each thread can handle different aspects of the computation, such as CPU-based preprocessing and GPU-based compute-intensive tasks, enabling efficient utilization of both CPU and GPU resources.
#include <iostream>
#include <vector>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 4

// CUDA kernel functions
__global__ void kernel1(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Perform computation on data
        data[tid] *= 2.0f;
    }
}

__global__ void kernel2(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Perform computation on data
        data[tid] += 1.0f;
    }
}

void threadFunc(int threadId, float* data, int size) {
    // Create CUDA context for the thread
    CUcontext context;
    CUdevice device;
    CUmodule module1, module2;
    CUfunction kernelFunc1, kernelFunc2;
    CUstream stream;
    CUevent startEvent1, endEvent1, startEvent2, endEvent2;
    CUresult cuResult;

    cuResult = cuInit(0);
    cuResult = cuDeviceGet(&device, 0);
    cuResult = cuCtxCreate(&context, 0, device);

    // Load CUDA modules and kernels
    cuResult = cuModuleLoad(&module1, "cuda_kernels1.ptx");
    cuResult = cuModuleLoad(&module2, "cuda_kernels2.ptx");
    cuResult = cuModuleGetFunction(&kernelFunc1, module1, "kernel1");
    cuResult = cuModuleGetFunction(&kernelFunc2, module2, "kernel2");

    // Create CUDA stream
    cuResult = cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    // Create CUDA events for synchronization
    cuResult = cuEventCreate(&startEvent1, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&endEvent1, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&startEvent2, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&endEvent2, CU_EVENT_DEFAULT);

    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Synchronize with the previous thread's end event
    if (threadId > 0) {
        cuResult = cuEventSynchronize(endEvent2);
    }

    // Record the start event for kernel1 in the current thread
    cuResult = cuEventRecord(startEvent1, stream);

    // Launch kernel1 in the current thread
    cuResult = cuLaunchKernel(kernelFunc1, gridSize, 1, 1, blockSize, 1, 1, 0, stream, &data, &size, nullptr);

    // Record the end event for kernel1 in the current thread
    cuResult = cuEventRecord(endEvent1, stream);

    // Synchronize with the next thread's start event
    if (threadId < NUM_THREADS - 1) {
        cuResult = cuEventSynchronize(startEvent2);
    }

    // Record the start event for kernel2 in the current thread
    cuResult = cuEventRecord(startEvent2, stream);

    // Launch kernel2 in the current thread
    cuResult = cuLaunchKernel(kernelFunc2, gridSize, 1, 1, blockSize, 1, 1, 0, stream, &data, &size, nullptr);

    // Record the end event for kernel2 in the current thread
    cuResult = cuEventRecord(endEvent2, stream);

    // Synchronize with the main thread
    cuResult = cuStreamSynchronize(stream);

    // Destroy CUDA events
    cuResult = cuEventDestroy(startEvent1);
    cuResult = cuEventDestroy(endEvent1);
    cuResult = cuEventDestroy(startEvent2);
    cuResult = cuEventDestroy(endEvent2);

    // Destroy CUDA stream
    cuResult = cuStreamDestroy(stream);

    // Destroy CUDA context
    cuResult = cuCtxDestroy(context);
}

int main() {
    int dataSize = 1024;
    std::vector<float> data(dataSize);

    // Initialize data
    for (int i = 0; i < dataSize; i++) {
        data[i] = static_cast<float>(i);
    }

    // Create multiple threads
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(threadFunc, i, data.data(), dataSize);
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    // Verify the result
    bool isCorrect = true;
    for (int i = 0; i < dataSize; i++) {
        if (data[i] != (2.0f * i + 1.0f)) {
            isCorrect = false;
            break;
        }
    }

    // Print the result
    if (isCorrect) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is incorrect!" << std::endl;
    }

    return 0;
}
