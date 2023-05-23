// 每个设备启动一个单独的线程。 每个线程处理其相应设备上的一部分数据

#include <cuda.h>
#include <iostream>
#include <vector>
#include <thread>

void processOnDevice(int deviceIndex, int portionSize, float* hostData) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr deviceData;

    cuDeviceGet(&device, deviceIndex);
    cuCtxCreate(&context, 0, device);
    (cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    (cuModuleGetFunction(&kernel, module, "_Z14vec_multiply_2Pfi"));
    cuMemAlloc(&deviceData, portionSize * sizeof(float));
    cuMemcpyHtoD(deviceData, hostData, portionSize * sizeof(float)); // Copy host data to device

    void* args[] = {&deviceData, &portionSize};
    cuLaunchKernel(kernel, (portionSize + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);

    cuCtxSynchronize();
    cuMemcpyDtoH(hostData, deviceData, portionSize * sizeof(float)); // Copy result back to host

    cuMemFree(deviceData);
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main() {
    CUresult result = cuInit(0);
    if(result != CUDA_SUCCESS) {
        std::cout << "Error initializing CUDA" << std::endl;
        return -1;
    }
    
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);

    const int dataSize = 16; // size of the data array
    const int portionSize = dataSize / deviceCount; // size of each portion
    float* hostData = new float[portionSize * deviceCount](); // Initialize host data to 0

    std::vector<std::thread> threads(deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        threads[i] = std::thread(processOnDevice, i, portionSize, hostData + i * portionSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Check results
    for (int i = 0; i < dataSize; i++) {
        if (hostData[i] != 0.0f) {
            std::cout << "Result verification failed at element " << i << ". Expected 0, got " << hostData[i] << std::endl;
            delete[] hostData;
            return -1;
        }
    }

    std::cout << "Result verification succeeded!" << std::endl;
    delete[] hostData;
    return 0;
}
