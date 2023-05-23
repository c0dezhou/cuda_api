#include <cuda.h>
#include <thread>
#include <vector>

#define dataSize 1024

std::vector<float> data(dataSize);
std::vector<float> results(dataSize);

void cpuComputation(std::vector<float>& data) {
    for (float& value : data) {
        // Perform some CPU computation here. We're just multiplying the value by 2.
        value *= 2.0f;
    }
}

void processOnDevice(CUdevice device, int deviceIndex) {
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr devData;
    CUdeviceptr devResults;
    int offset = deviceIndex * dataSize / 2;

    CUDA_CALL(cuCtxCreate(&context, 0, device));
    CUDA_CALL(cuModuleLoad(&module, "double.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernel, module, "_Z6doublePfS_ii"));

    CUDA_CALL(cuMemAlloc(&devData, dataSize * sizeof(float) / 2));
    CUDA_CALL(cuMemAlloc(&devResults, dataSize * sizeof(float) / 2));

    CUDA_CALL(cuMemcpyHtoD(devData, &data[offset], dataSize * sizeof(float) / 2));

    void* args[] = { &devData, &devResults, &(dataSize / 2) };
    CUDA_CALL(cuLaunchKernel(kernel, dataSize / 2, 1, 1, 1, 1, 1, 0, 0, args, 0));
    
    CUDA_CALL(cuMemcpyDtoH(&results[offset], devResults, dataSize * sizeof(float) / 2));

    CUDA_CALL(cuMemFree(devData));
    CUDA_CALL(cuMemFree(devResults));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    CUDA_CALL(cuInit(0));

    CUdevice device0, device1;
    CUDA_CALL(cuDeviceGet(&device0, 0));
    CUDA_CALL(cuDeviceGet(&device1, 1));

    // Initialize data
    for (int i = 0; i < dataSize; i++) {
        data[i] = static_cast<float>(i);
    }

    std::thread cpuThread(cpuComputation, std::ref(data));

    std::thread processThread0(processOnDevice, device0, 0);
    std::thread processThread1(processOnDevice, device1, 1);

    cpuThread.join();
    processThread0.join();
    processThread1.join();

    // check results
    for (int i = 0; i < dataSize; i++) {
        if (results[i] != 4.0f * i) {
            std::cout << "Error: result mismatch at position " << i << std::endl;
            return -1;
        }
    }

    std::cout << "All results are correct." << std::endl;

    return 0;
}
