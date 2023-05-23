#include <cuda.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>
#include <cmath>
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

struct Task {
    CUdeviceptr deviceData;
    int dataSize;
};

std::mutex mtx;
std::condition_variable cv;
std::queue<Task> taskQueue;
std::atomic<bool> stop(false);

void prepareData(CUcontext context, int deviceIndex) {
    while (!stop) {
        CUdeviceptr deviceData;
        int dataSize = 1024;  // Here, we're just always using a fixed size for simplicity.
        CUDA_CALL(cuMemAlloc(&deviceData, dataSize * sizeof(float)));

        std::vector<float> hostData(dataSize, 32);
        // Here, fill hostData with the actual data you want to process.

        CUDA_CALL(cuMemcpyHtoD(deviceData, hostData.data(), dataSize * sizeof(float)));

        std::unique_lock<std::mutex> lock(mtx);
        taskQueue.push({deviceData, dataSize});
        lock.unlock();

        cv.notify_one();
    }
}

void processData(CUcontext context, CUmodule module) {
    while (!stop || !taskQueue.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !taskQueue.empty(); });

        Task task = taskQueue.front();
        taskQueue.pop();

        lock.unlock();

        CUfunction kernelFunc;
        CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "kernel_func_name"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        CUDA_CALL(cuCtxSynchronize());
    }
}

void retrieveData(CUcontext context, int deviceIndex) {
    while (!stop || !taskQueue.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !taskQueue.empty(); });

        Task task = taskQueue.front();
        taskQueue.pop();

        lock.unlock();

        std::vector<float> hostData(task.dataSize);
        CUDA_CALL(cuMemcpyDtoH(hostData.data(), task.deviceData, task.dataSize * sizeof(float)));
        CUDA_CALL(cuMemFree(task.deviceData));

        // Print the data for simplicity.
        for (float val : hostData) {
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {
    cuInit(0);

    CUdevice device0, device1;
    CUDA_CALL(cuDeviceGet(&device0, 0));
    CUDA_CALL(cuDeviceGet(&device1, 1));

    CUcontext context0, context1;
    CUDA_CALL(cuCtxCreate(&context0, 0, device0));
    CUDA_CALL(cuCtxCreate(&context1, 0, device1));

    CUmodule module0, module1;
    CUDA_CALL(cuModuleLoad(&module0, "path_to_your_ptx_file_for_device_0.ptx"));
    CUDA_CALL(cuModuleLoad(&module1, "path_to_your_ptx_file_for_device_1.ptx"));

    std::thread prepareThread0(prepareData, context0, 0);
    std::thread processThread0(processData, context0, module0);
    std::thread retrieveThread0(retrieveData, context0, 0);

    std::thread prepareThread1(prepareData, context1, 1);
    std::thread processThread1(processData, context1, module1);
    std::thread retrieveThread1(retrieveData, context1, 1);

    // Wait for some condition here, then set stop to true to stop the threads.
    stop = true;

    prepareThread0.join();
    processThread0.join();
    retrieveThread0.join();

    prepareThread1.join();
    processThread1.join();
    retrieveThread1.join();

    CUDA_CALL(cuModuleUnload(module0));
    CUDA_CALL(cuCtxDestroy(context0));

    CUDA_CALL(cuModuleUnload(module1));
    CUDA_CALL(cuCtxDestroy(context1));

    return 0;
}
