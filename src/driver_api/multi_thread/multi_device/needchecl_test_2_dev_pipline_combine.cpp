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

#define checkError(op) __checkError_error((op), #op, __FILE__, __LINE__)
bool __checkError_error(CUresult code,
                        const char* op,
                        const char* file,
                        int line) {
    if (code != CUresult::CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op,
               err_name, err_message);
        return false;
    }
    return true;
}

struct Task {
    CUdeviceptr deviceData;
    int dataSize;
};

std::mutex mtx1, mtx2;
std::condition_variable cv1, cv2;
std::queue<Task> taskQueue1, taskQueue2;
std::atomic<bool> stop(false);

void prepareData(CUcontext context, int deviceIndex) {
    checkError(cuCtxSetCurrent(context));
    while (!stop) {
        CUdeviceptr deviceData;
        int dataSize = 1024;  // Here, we're just always using a fixed size for simplicity.
        checkError(cuMemAlloc(&deviceData, dataSize * sizeof(float)));

        std::vector<float> hostData(dataSize);
        // Here, fill hostData with the actual data you want to process.

        checkError(cuMemcpyHtoD(deviceData, hostData.data(), dataSize * sizeof(float)));

        std::unique_lock<std::mutex> lock(mtx1);
        taskQueue1.push({deviceData, dataSize});
        lock.unlock();

        cv1.notify_one();
    }
}

void processDataOnDevice1(CUcontext context) {
    checkError(cuCtxSetCurrent(context));
    while (!stop || !taskQueue1.empty()) {
        std::unique_lock<std::mutex> lock1(mtx1);
        cv1.wait(lock1, []{ return !taskQueue1.empty(); });

        Task task = taskQueue1.front();
        taskQueue1.pop();

        lock1.unlock();

        CUmodule module0;
        checkError(
            cuModuleLoad(&module0,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "cuda_kernel.ptx"));
        CUfunction kernelFunc;
        checkError(
            cuModuleGetFunction(&kernelFunc, module0, "_Z14vec_multiply_2Pfi"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        checkError(cuCtxSynchronize());

        // Transfer the data to the second device queue
        std::unique_lock<std::mutex> lock2(mtx2);
        taskQueue2.push(task);
        lock2.unlock();

        cv2.notify_one();
        checkError(cuModuleUnload(module0));
    }
}

void processDataOnDevice2(CUcontext context) {
    checkError(cuCtxSetCurrent(context));
    while (!stop || !taskQueue2.empty()) {
        std::unique_lock<std::mutex> lock(mtx2);
        cv2.wait(lock, []{ return !taskQueue2.empty(); });

        Task task = taskQueue2.front();
        taskQueue2.pop();

        lock.unlock();

        CUmodule module0;
        checkError(
            cuModuleLoad(&module0,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "cuda_kernel.ptx"));

        CUfunction kernelFunc;
        checkError(cuModuleGetFunction(&kernelFunc, module0, "_Z9vec_add_3Pfi"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        checkError(cuCtxSynchronize());

        std::vector<float> hostData(task.dataSize);
        checkError(cuMemcpyDtoH(hostData.data(), task.deviceData, task.dataSize * sizeof(float)));

        // Here, you can do something with hostData.
        for (auto i = 0; i < hostData.size();i++) {
            std::cout << hostData[i] << std::endl;
        }

        checkError(cuMemFree(task.deviceData));
        checkError(cuModuleUnload(module0));
    }
}

int main() {
    cuInit(0);

    CUdevice device0, device1;
    checkError(cuDeviceGet(&device0, 0));
    checkError(cuDeviceGet(&device1, 1));

    CUcontext context0, context1;
    checkError(cuCtxCreate(&context0, 0, device0));
    checkError(cuCtxCreate(&context1, 0, device1));

    // CUmodule module0, module1;
    // checkError(cuModuleLoad(&module0,
    //                        "/data/system/yunfan/cuda_api/common/cuda_kernel/"
    //                        "cuda_kernel.ptx"));
    // checkError(cuModuleLoad(&module1,
    //                        "/data/system/yunfan/cuda_api/common/cuda_kernel/"
    //                        "cuda_kernel.ptx"));

    std::thread prepareThread(prepareData, context0, 0);
    std::thread processThread0(processDataOnDevice1, context0);
    std::thread processThread1(processDataOnDevice2, context1);

    // Wait for some condition here, then set stop to true to stop the threads.
    stop = true;

    prepareThread.join();
    processThread0.join();
    processThread1.join();

    
    checkError(cuCtxDestroy(context0));
    checkError(cuCtxDestroy(context1));

    return 0;
}

