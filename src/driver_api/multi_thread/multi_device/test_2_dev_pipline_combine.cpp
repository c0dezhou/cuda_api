// 第一个设备对数据执行一些计算并将结果写回主机内存。 然后将这些结果用作第二个设备的输入。

// 这种方法需要同步管理两个设备的两个线程，以确保第一个设备在第二个设备开始之前完成计算。

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

std::mutex mtx1, mtx2;
std::condition_variable cv1, cv2;
std::queue<Task> taskQueue1, taskQueue2;
std::atomic<bool> stop(false);

void prepareData(CUcontext context, int deviceIndex) {
    while (!stop) {
        CUdeviceptr deviceData;
        int dataSize = 1024;  // Here, we're just always using a fixed size for simplicity.
        CUDA_CALL(cuMemAlloc(&deviceData, dataSize * sizeof(float)));

        std::vector<float> hostData(dataSize);
        // Here, fill hostData with the actual data you want to process.

        CUDA_CALL(cuMemcpyHtoD(deviceData, hostData.data(), dataSize * sizeof(float)));

        std::unique_lock<std::mutex> lock(mtx1);
        taskQueue1.push({deviceData, dataSize});
        lock.unlock();

        cv1.notify_one();
    }
}

void processDataOnDevice1(CUcontext context, CUmodule module) {
    while (!stop || !taskQueue1.empty()) {
        std::unique_lock<std::mutex> lock1(mtx1);
        cv1.wait(lock1, []{ return !taskQueue1.empty(); });

        Task task = taskQueue1.front();
        taskQueue1.pop();

        lock1.unlock();

        CUfunction kernelFunc;
        CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "kernel_func_name"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        CUDA_CALL(cuCtxSynchronize());

        // Transfer the data to the second device queue
        std::unique_lock<std::mutex> lock2(mtx2);
        taskQueue2.push(task);
        lock2.unlock();

        cv2.notify_one();
    }
}

void processDataOnDevice2(CUcontext context, CUmodule module) {
    while (!stop || !taskQueue2.empty()) {
        std::unique_lock<std::mutex> lock(mtx2);
        cv2.wait(lock, []{ return !taskQueue2.empty(); });

        Task task = taskQueue2.front();
        taskQueue2.pop();

        lock.unlock();

        CUfunction kernelFunc;
        CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "kernel_func_name"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        CUDA_CALL(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        CUDA_CALL(cuCtxSynchronize());

        std::vector<float> hostData(task.dataSize);
        CUDA_CALL(cuMemcpyDtoH(hostData.data(), task.deviceData, task.dataSize * sizeof(float)));

        // Here, you can do something with hostData.

        CUDA_CALL(cuMemFree(task.deviceData));
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

    std::thread prepareThread(prepareData, context0, 0);
    std::thread processThread0(processDataOnDevice1, context0, module0);
    std::thread processThread1(processDataOnDevice2, context1, module1);

    // Wait for some condition here, then set stop to true to stop the threads.
    stop = true;

    prepareThread.join();
    processThread0.join();
    processThread1.join();

    CUDA_CALL(cuModuleUnload(module0));
    CUDA_CALL(cuCtxDestroy(context0));

    CUDA_CALL(cuModuleUnload(module1));
    CUDA_CALL(cuCtxDestroy(context1));

    return 0;
}

