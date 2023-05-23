#include <iostream>
#include <vector>
#include <thread>
#include <cuda.h>

// 此代码创建一个 CUDA 上下文，加载包含内核函数的 CUDA 模块，并检索 kernel1 和 kernel2 的函数指针。 然后启动多个线程，其中每个线程执行 launchKernels 函数。 launchKernels 函数为每个线程设置 CUDA 上下文，并根据线程 ID 启动 kernel1 或 kernel2。

// CUDA kernel 1
__global__ void kernel1(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = data[tid] * 2;
}

// CUDA kernel 2
__global__ void kernel2(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = data[tid] + 10;
}

// CUDA error checking macro
#define CUDA_CHECK_ERROR(func) \
    do { \
        CUresult result = (func); \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(result, &errorStr); \
            std::cerr << "CUDA error: " << errorStr << std::endl; \
            exit(1); \
        } \
    } while (0)

void launchKernels(CUcontext context, CUfunction kernel1Func, CUfunction kernel2Func, float* d_data, int dataSize) {
    // Get the thread ID
    int threadId = std::stoi(std::this_thread::get_id().hash());

    // Calculate the portion of data for this thread
    int numThreads = std::thread::hardware_concurrency();
    int dataSizePerThread = dataSize / numThreads;
    int startIndex = threadId * dataSizePerThread;
    int endIndex = startIndex + dataSizePerThread;

    if (threadId == numThreads - 1) {
        // Last thread takes care of the remaining elements if size % numThreads != 0
        endIndex += dataSize % numThreads;
    }

    // Set up the CUDA context for the thread
    CUDA_CHECK_ERROR(cuCtxSetCurrent(context));

    // Launch kernel1
    if (threadId % 2 == 0) {
        CUDA_CHECK_ERROR(cuLaunchKernel(kernel1Func, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)&d_data, NULL));
    }

    // Launch kernel2
    if (threadId % 2 == 1) {
        CUDA_CHECK_ERROR(cuLaunchKernel(kernel2Func, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)&d_data, NULL));
    }
}

int main() {
    // Initialize CUDA
    CUDA_CHECK_ERROR(cuInit(0));

    // Create CUDA context
    CUcontext context;
    CUDA_CHECK_ERROR(cuCtxCreate(&context, 0, 0));

    // Load CUDA module
    CUmodule module;
    CUDA_CHECK_ERROR(cuModuleLoad(&module, "kernel.ptx"));

    // Get CUDA kernel functions
    CUfunction kernel1Func, kernel2Func;
    CUDA_CHECK_ERROR(cuModuleGetFunction(&kernel1Func, module, "kernel1"));
    CUDA_CHECK_ERROR(cuModuleGetFunction(&kernel2Func, module, "kernel2"));

    // Data size
    int dataSize = 1024;

    // Allocate device memory
    CUdeviceptr d_data;
    CUDA_CHECK_ERROR(cuMemAlloc(&d_data, dataSize * sizeof(float)));

    // Launch multiple threads
    std::vector<std::thread> threads;

    for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
        threads.emplace_back(launchKernels, context, kernel1Func, kernel2Func, (float*)d_data, dataSize);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Copy device memory to host
    float* h_data = new float[dataSize];
    CUDA_CHECK_ERROR(cuMemcpyDtoH(h_data, d_data, dataSize * sizeof(float)));

    // Print the results
    for (int i = 0; i < dataSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup and release resources
    delete[] h_data;
    CUDA_CHECK_ERROR(cuMemFree(d_data));
    CUDA_CHECK_ERROR(cuModuleUnload(module));
    CUDA_CHECK_ERROR(cuCtxDestroy(context));

    return 0;
}
