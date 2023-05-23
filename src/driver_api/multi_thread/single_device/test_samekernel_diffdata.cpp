#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cassert>

// CUDA kernel
__global__ void vecAdd(float* A, float* B, float* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

// Result verification
bool verifyResult(const std::vector<float>& h_A, const std::vector<float>& h_B, const std::vector<float>& h_C) {
    for (size_t i = 0; i < h_C.size(); ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            std::cout << "Verification failed at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

// CUDA operation performed by each thread
void cudaOperation(int threadId, CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int size) {
    // Set CUDA device
    cuCtxSetCurrent(NULL);

    // Initialize CUDA context
    CUcontext context;
    cuCtxCreate(&context, 0, 0);

    // Allocate memory on the device
    CUresult result;
    result = cuMemAlloc(&d_A, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Initialize input data for this thread
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    for (int i = 0; i < size; ++i) {
        h_A[i] = i + threadId;
        h_B[i] = i - threadId;
    }

    // Copy input data from host to device
    result = cuMemcpyHtoD(d_A, h_A.data(), size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemcpyHtoD(d_B, h_B.data(), size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Launch the CUDA kernel
    CUmodule module;
    result = cuModuleLoad(&module, "cuda_kernel.ptx");
    assert(result == CUDA_SUCCESS);

    CUfunction function;
    result = cuModuleGetFunction(&function, module, "vecAdd");
    assert(result == CUDA_SUCCESS);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    void* kernelParams[] = { &d_A, &d_B, &d_C, &size };
    result = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, kernelParams, NULL);
    assert(result == CUDA_SUCCESS);

    // Copy the result back from the device to host
    std::vector<float> h_C(size);
    result = cuMemcpyDtoH(h_C.data(), d_C, size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Verify the result
    bool result = verifyResult(h_A, h_B, h_C);
    assert(result);

    // Free device memory
    result = cuMemFree(d_A);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_B);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_C);
    assert(result == CUDA_SUCCESS);

    // Destroy CUDA context
    cuCtxDestroy(context);
}

int main() {
    // Initialize CUDA driver API
    cuInit(0);

    const int numThreads = 20; // Number of threads
    const int dataSize = 1000; // Size of data per thread

    std::vector<std::thread> threads(numThreads);
    std::vector<CUdeviceptr> d_A(numThreads);
    std::vector<CUdeviceptr> d_B(numThreads);
    std::vector<CUdeviceptr> d_C(numThreads);

    // Launch CUDA operations in multiple threads
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(cudaOperation, i, d_A[i], d_B[i], d_C[i], dataSize);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "All threads finished executing CUDA operations." << std::endl;

    return 0;
}
