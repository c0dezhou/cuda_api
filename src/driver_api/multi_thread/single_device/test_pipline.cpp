// 其中一个线程为下一次 GPU 计算准备数据，另一个线程启动内核，第三个线程检索结果。 这种方法允许隐藏数据传输和内核启动的延迟，并让 GPU 保持忙碌。
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>

// 互斥锁是多线程编程中用于同步线程的工具。 它用于保护共享数据不被多个线程并发访问。 线程必须先锁定互斥体才能访问共享数据。 如果另一个线程已经锁定了互斥量，则该线程将阻塞直到互斥量被解锁。

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

#define NUM_THREADS 3
#define N 1024

std::atomic<bool> dataReady(false);
std::atomic<bool> computationDone(false);
std::condition_variable cv;
std::mutex cv_m;

void prepareData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }

    {
        std::lock_guard<std::mutex> lk(cv_m);
        dataReady = true;
    }
    cv.notify_all();
}

void retrieveResults(float* data, int size) {
    std::unique_lock<std::mutex> lk(cv_m);
    cv.wait(lk, []{ return computationDone.load(); });

    // Checking results
    for (int i = 0; i < size; ++i) {
        if (data[i] != 2 * i) {
            std::cerr << "Check failed at index " << i << "! Expected " << 2 * i << ", got " << data[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Check passed!" << std::endl;
}

void launchKernel(float* data, int size) {
    std::unique_lock<std::mutex> lk(cv_m);
    cv.wait(lk, []{ return dataReady.load(); });

    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;
    CUdeviceptr deviceData;

    CUDA_CALL(cuDeviceGet(&device, 0));
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z10add_kernelPfS_S_i"));

    CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    // Allocate device memory and copy from host to device
    CUDA_CALL(cuMemAlloc(&deviceData, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoDAsync(deviceData, data, size * sizeof(float), stream));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void* args[] = { &deviceData, &deviceData, &deviceData, &size };
    CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr));
    CUDA_CALL(cuStreamSynchronize(stream));

    // Copy from device to host
    CUDA_CALL(cuMemcpyDtoHAsync(data, deviceData, size * sizeof(float), stream));
    CUDA_CALL(cuStreamSynchronize(stream));

    CUDA_CALL(cuStreamDestroy(stream));
    CUDA_CALL(cuCtxDestroy(context));

    computationDone = true;
    cv.notify_all();
}

int main() {
    CUDA_CALL(cuInit(0));

    float* data = new float[N];

    std::thread preparer(prepareData, data, N);
    std::thread launcher(launchKernel, data, N);
    std::thread retriever(retrieveResults, data, N);

    preparer.join();
    launcher.join();
    retriever.join();

    delete[] data;

    return 0;
}
