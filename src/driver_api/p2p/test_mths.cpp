#include <cuda.h>
#include <iostream>
#include <vector>

#define dataSize 1024

std::vector<float> results(dataSize);

void processOnDevice(CUdevice device, int deviceIndex, CUdeviceptr devData) {
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr devResults;
    int offset = deviceIndex * dataSize / 2;

    CUDA_CALL(cuCtxCreate(&context, 0, device));
    CUDA_CALL(cuModuleLoad(&module, "double.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernel, module, "_Z6doublePfS_ii"));

    CUDA_CALL(cuMemAlloc(&devResults, dataSize * sizeof(float) / 2));

    void* args[] = { &devData, &devResults, &(dataSize / 2) };
    CUDA_CALL(cuLaunchKernel(kernel, dataSize / 2, 1, 1, 1, 1, 1, 0, 0, args, 0));
    
    CUDA_CALL(cuMemcpyDtoH(&results[offset], devResults, dataSize * sizeof(float) / 2));

    CUDA_CALL(cuMemFree(devResults));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    CUDA_CALL(cuInit(0));

    CUdevice device0, device1;
    CUDA_CALL(cuDeviceGet(&device0, 0));
    CUDA_CALL(cuDeviceGet(&device1, 1));

    CUcontext context0, context1;
    CUDA_CALL(cuCtxCreate(&context0, 0, device0));
    CUDA_CALL(cuCtxCreate(&context1, 0, device1));

    CUdeviceptr devData;
    CUDA_CALL(cuMemAlloc(&devData, dataSize * sizeof(float)));
    
    // Enable P2P access from device1 to device0
    CUDA_CALL(cuCtxSetCurrent(context1));
    CUDA_CALL(cuDeviceEnablePeerAccess(device0, 0));

    std::thread processThread0(processOnDevice, device0, 0, devData);
    std::thread processThread1(processOnDevice, device1, 1, devData);

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

    CUDA_CALL(cuMemFree(devData));

    return 0;
}
