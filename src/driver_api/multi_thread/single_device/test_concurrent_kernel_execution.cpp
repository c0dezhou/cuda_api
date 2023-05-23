#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>

#define NUM_THREADS 2

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

void threadFunc(int threadId, float* hostData, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;
    CUdeviceptr deviceData;

    CUDA_CALL(cuDeviceGet(&device, 0));
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    if(threadId == 0)
        CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z14vec_multiply_2Pfi"));
    else
        CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z9vec_add_3Pfi"));

    CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    // Allocate device memory and copy from host to device
    CUDA_CALL(cuMemAlloc(&deviceData, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoD(deviceData, hostData, size * sizeof(float)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void * args[] = {&deviceData, &size};
    CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr));
    CUDA_CALL(cuStreamSynchronize(stream));

    // Copy results from device to host
    CUDA_CALL(cuMemcpyDtoH(hostData, deviceData, size * sizeof(float)));

    CUDA_CALL(cuStreamDestroy(stream));
    CUDA_CALL(cuMemFree(deviceData));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    CUDA_CALL(cuInit(0));
    int dataSize = 1024 * NUM_THREADS;
    std::vector<float> data(dataSize, 0.0f);

    std::vector<std::thread> threads(NUM_THREADS);
    int chunkSize = dataSize / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(threadFunc, i, data.data() + i * chunkSize, chunkSize);
    }

    for (auto &t : threads) {
        t.join();
    }

    for (const auto &value : data) {
        std::cout << value << std::endl;
    }

    return 0;
}
