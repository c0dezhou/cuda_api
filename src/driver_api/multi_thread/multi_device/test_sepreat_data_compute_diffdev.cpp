#include <cuda.h>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <atomic>

#define CUDA_CALL(call) \
do { \
    CUresult cuResult = call; \
    if (cuResult != CUDA_SUCCESS) { \
        const char *errName = nullptr, *errStr = nullptr; \
        cuGetErrorName(cuResult, &errStr); \
        cuGetErrorString(cuResult, &errName); \
        std::cerr << "CUDA error: " << errName << " (" << errStr << ") at " << #call << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Replace with your kernel function name
#define KERNEL_FUNC_NAME "_Z14vec_multiply_2Pfi"

struct Task {
    CUdeviceptr deviceData;
    int dataSize;
};

std::mutex mtx;
std::vector<float> results;

void processOnDevice(CUdevice device, int deviceIndex) {
    CUcontext context;
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    CUmodule module;
    CUDA_CALL(cuModuleLoad(&module, "kernel.ptx"));

    CUfunction kernelFunc;
    CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, KERNEL_FUNC_NAME));

    int dataSize = 1024; // Adjust based on your problem size
    CUdeviceptr deviceData;
    CUDA_CALL(cuMemAlloc(&deviceData, dataSize * sizeof(float)));

    std::vector<float> hostData(dataSize);
    // Fill hostData with the actual data you want to process.

    CUDA_CALL(cuMemcpyHtoD(deviceData, hostData.data(), dataSize * sizeof(float)));

    int blockSize = 256;
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    void* args[] = {&deviceData, &dataSize};
    CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
    CUDA_CALL(cuCtxSynchronize());

    std::vector<float> resultData(dataSize);
    CUDA_CALL(cuMemcpyDtoH(resultData.data(), deviceData, dataSize * sizeof(float)));

    std::lock_guard<std::mutex> lock(mtx);
    results.insert(results.end(), resultData.begin(), resultData.end());

    CUDA_CALL(cuMemFree(deviceData));
    CUDA_CALL(cuModuleUnload(module));
    CUDA_CALL(cuCtxDestroy(context));
}

bool checkResults(const std::vector<float>& results, const std::vector<float>& originalData) {
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] != 2 * originalData[i]) {
            std::cerr << "Result verification failed at element " << i << ". Expected " << (2 * originalData[i]) << ", but got " << results[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    cuInit(0);

    CUdevice device0, device1;
    CUDA_CALL(cuDeviceGet(&device0, 0));
    CUDA_CALL(cuDeviceGet(&device1, 1));

    std::thread processThread0(processOnDevice, device0, 0);
    std::thread processThread1(processOnDevice, device1, 1);

    processThread0.join();
    processThread1.join();

    std::vector<float> originalData(2 * dataSize);
    // Fill originalData with your initial data.
    
    if (checkResults(results, originalData)) {
        std::cout << "Result verification passed." << std::endl;
    } else {
        std::cerr << "Result verification failed." << std::endl;
    }

    return 0;
}

