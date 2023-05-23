// 将主机和设备之间的数据传输与内核执行重叠

#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>

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

#define NUM_THREADS 2
#define N 1024

void threadFunc(int threadId, float* hostA, float* hostB, float* hostC, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;
    CUdeviceptr deviceA, deviceB, deviceC;

    CUDA_CALL(cuDeviceGet(&device, 0));
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z10add_kernelPfS_S_i"));

    CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    // Allocate device memory and copy from host to device
    CUDA_CALL(cuMemAlloc(&deviceA, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoDAsync(deviceA, hostA, size * sizeof(float), stream));

    CUDA_CALL(cuMemAlloc(&deviceB, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoDAsync(deviceB, hostB, size * sizeof(float), stream));

    CUDA_CALL(cuMemAlloc(&deviceC, size * sizeof(float)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void * args[] = {&deviceA, &deviceB, &deviceC, &size};
    CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr));

    // Copy results from device to host
    CUDA_CALL(cuMemcpyDtoHAsync(hostC, deviceC, size * sizeof(float), stream));
    CUDA_CALL(cuStreamSynchronize(stream));

    CUDA_CALL(cuStreamDestroy(stream));
    CUDA_CALL(cuMemFree(deviceA));
    CUDA_CALL(cuMemFree(deviceB));
    CUDA_CALL(cuMemFree(deviceC));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    CUDA_CALL(cuInit(0));

    float *hostA, *hostB, *hostC;
    hostA = new float[N];
    hostB = new float[N];
    hostC = new float[N];

    for (int i = 0; i < N; ++i) {
        hostA[i] = i;
        hostB[i] = i;
    }

    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i] = std::thread(threadFunc, i, hostA, hostB, hostC, N);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Checking results
    for (int i = 0; i < N; ++i) {
        if (hostC[i] != 2 * i) {
            std::cerr << "Check failed at index " << i << "! Expected " << 2 * i << ", got " << hostC[i] << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Check passed!" << std::endl;

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;

    return 0;
}
