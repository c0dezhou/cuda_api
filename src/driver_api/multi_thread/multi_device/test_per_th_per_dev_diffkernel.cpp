#include <cuda.h>
#include <iostream>
#include <vector>
#include <thread>

const char* kernelSources[] = {
    "__global__ void multiplyKernel(float *data, int size) {\n"
    "    int idx = threadIdx.x + blockIdx.x * blockDim.x;\n"
    "    if (idx < size) {\n"
    "        data[idx] *= 2.0f;\n" // Multiply each element by 2
    "    }\n"
    "}\n",
    "__global__ void addKernel(float *data, int size) {\n"
    "    int idx = threadIdx.x + blockIdx.x * blockDim.x;\n"
    "    if (idx < size) {\n"
    "        data[idx] += 1.0f;\n" // Add 1 to each element
    "    }\n"
    "}\n"
};

#define CUDA_CALL(call) \
do { \
    CUresult cuResult = call; \
    if (cuResult != CUDA_SUCCESS) { \
        const char *errName = nullptr, *errStr = nullptr; \
        cuGetErrorName(cuResult, &errName); \
        cuGetErrorString(cuResult, &errStr); \
        std::cerr << "CUDA error: " << errName << " (" << errStr << ") at " << #call << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void processOnDevice(int deviceIndex, int portionSize, float* hostData) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr deviceData;

    cuDeviceGet(&device, deviceIndex);
    cuCtxCreate(&context, 0, device);
    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    if(deviceIndex == 0)
        CUDA_CALL(cuModuleGetFunction(&kernel, module, "_Z14vec_multiply_2Pfi"));
    else
        CUDA_CALL(cuModuleGetFunction(&kernel, module, "_Z9vec_add_3Pfi"));

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

    const int dataSize = 1e6; // size of the data array
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
        float expected = (i < dataSize / 2) ? 0.0f : 1.0f; // First half should be 0, second half should be 1
        if (hostData[i] != expected) {
            std::cout << "Result verification failed at element " << i << ". Expected " << expected << ", got " << hostData[i] << std::endl;
            delete[] hostData;
            return -1;
        }
    }

    std::cout << "Result verification succeeded!" << std::endl;
    delete[] hostData;
    return 0;
}

// i think that you deliver the deviceIndex to be a param, did you want use it to distinguish different device?