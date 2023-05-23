#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>

#define NUM_THREADS 4

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

void cpuComputation(std::vector<float>& data) {
    for (float& value : data) {
        // Perform some CPU computation here. For example, we're just multiplying the value by 2.
        value *= 2.0f;
    }
}

void threadFunc(int threadId, std::vector<float>& data, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;

    CUDA_CALL(cuDeviceGet(&device, 0));
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z9vec_sub_1Pfi"));

    CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    cpuComputation(data);  // Perform some CPU computation before the GPU computation.

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    CUdeviceptr d_data;
    size_t bytes = size * sizeof(float);
    CUDA_CALL(cuMemAlloc(&d_data, bytes));
    CUDA_CALL(cuMemcpyHtoD(d_data, data.data(), bytes));

    void * args[] = {&d_data, &size};
    CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr));
    CUDA_CALL(cuStreamSynchronize(stream));

    CUDA_CALL(cuMemcpyDtoH(data.data(), d_data, bytes));
    CUDA_CALL(cuMemFree(d_data));

    CUDA_CALL(cuStreamDestroy(stream));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    CUDA_CALL(cuInit(0));
    int dataSize = 10;
    std::vector<float> data(dataSize, 0.0f);

    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(threadFunc, i, std::ref(data), dataSize); // std::ref 通常用于传递对可复制对象的引用
    }

    for (auto &t : threads) {
        t.join();
    }

    for (const auto &value : data) {
        std::cout << value << std::endl;
    }

    return 0;
}
