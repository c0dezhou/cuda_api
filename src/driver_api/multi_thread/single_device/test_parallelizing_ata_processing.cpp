// Parallelizing Data Processing: You can divide a large dataset into smaller portions and assign each portion to a separate thread. Each thread can independently process its assigned data, allowing for parallel processing and improved performance. This approach is commonly used in tasks such as image/video processing, signal processing, and simulations.
#include <iostream>
#include <vector>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 4
#define DATA_SIZE 1024

// CUDA kernel function
__global__ void processData(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Process data element
        data[tid] *= 2.0f;
    }
}

void processThread(float* data, int startIndex, int endIndex) {
    // Create CUDA context for the thread
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernel;
    CUresult cuResult;

    cuResult = cuInit(0);
    cuResult = cuDeviceGet(&device, 0);
    cuResult = cuCtxCreate(&context, 0, device);
    
    // Load CUDA module and kernel
    cuResult = cuModuleLoad(&module, "cuda_kernel.ptx");
    cuResult = cuModuleGetFunction(&kernel, module, "processData");

    // Allocate and copy data on the device
    float* d_data;
    cuResult = cuMemAlloc(&d_data, DATA_SIZE * sizeof(float));
    cuResult = cuMemcpyHtoD(d_data, data, DATA_SIZE * sizeof(float));

    // Set up kernel parameters
    void* args[] = { &d_data, &DATA_SIZE };
    size_t argSizes[] = { sizeof(CUdeviceptr), sizeof(int) };

    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (DATA_SIZE + blockSize - 1) / blockSize;

    // Launch the kernel
    cuResult = cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, argSizes);

    // Synchronize device
    cuResult = cuCtxSynchronize();

    // Copy data back to host
    cuResult = cuMemcpyDtoH(data + startIndex, d_data + startIndex, (endIndex - startIndex) * sizeof(float));

    // Clean up resources
    cuResult = cuMemFree(d_data);
    cuResult = cuModuleUnload(module);
    cuResult = cuCtxDestroy(context);
}

int main() {
    // Initialize data
    std::vector<float> data(DATA_SIZE, 1.0f);

    // Create threads
    std::vector<std::thread> threads(NUM_THREADS);
    int dataSizePerThread = DATA_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int startIndex = i * dataSizePerThread;
        int endIndex = startIndex + dataSizePerThread;
        if (i == NUM_THREADS - 1) {
            // Last thread takes care of the remaining elements if DATA_SIZE % NUM_THREADS != 0
            endIndex += DATA_SIZE % NUM_THREADS;
        }

        threads[i] = std::thread(processThread, data.data(), startIndex, endIndex);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify the result
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (data[i] != 2.0f) {
            std::cout << "Incorrect result at index " << i << std::endl;
            break;
        }
    }

    std::cout << "Result verification complete." << std::endl;

    return 0;
}
