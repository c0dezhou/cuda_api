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

std::mutex mtx;
std::condition_variable cv;
std::queue<Task> taskQueue;
std::atomic<bool> stop(false);

void prepareData(CUcontext context, int deviceIndex) {
    checkError(cuCtxSetCurrent(context));
    while (!stop) {
        CUdeviceptr deviceData;
        int dataSize = 1024;  // Here, we're just always using a fixed size for simplicity.
        checkError(cuMemAlloc(&deviceData, dataSize * sizeof(float)));

        std::vector<float> hostData(dataSize, 32);
        // Here, fill hostData with the actual data you want to process.

        checkError(cuMemcpyHtoD(deviceData, hostData.data(), dataSize * sizeof(float)));

        std::unique_lock<std::mutex> lock(mtx);
        taskQueue.push({deviceData, dataSize});
        lock.unlock();

        cv.notify_one();
    }
}

void processData(CUcontext context) {
    checkError(cuCtxSetCurrent(context));
    while (!stop || !taskQueue.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !taskQueue.empty(); });

        Task task = taskQueue.front();
        taskQueue.pop();

        lock.unlock();

        CUmodule module;
        checkError(
            cuModuleLoad(&module,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "cuda_kernel.ptx"));
        CUfunction kernelFunc;
        checkError(cuModuleGetFunction(&kernelFunc, module, "_Z9vec_add_3Pfi"));

        int blockSize = 256;
        int gridSize = (task.dataSize + blockSize - 1) / blockSize;

        void* args[] = {&task.deviceData, &task.dataSize};
        checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr));
        checkError(cuCtxSynchronize());
        checkError(cuModuleUnload(module));
    }
}

void retrieveData(CUcontext context, int deviceIndex) {
    checkError(cuCtxSetCurrent(context));
    while (!stop || !taskQueue.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !taskQueue.empty(); });

        Task task = taskQueue.front();
        taskQueue.pop();

        lock.unlock();

        std::vector<float> hostData(task.dataSize);
        checkError(cuMemcpyDtoH(hostData.data(), task.deviceData, task.dataSize * sizeof(float)));
        checkError(cuMemFree(task.deviceData));

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

    std::thread prepareThread0(prepareData, context0, 0);
    std::thread processThread0(processData, context0);
    std::thread retrieveThread0(retrieveData, context0, 0);

    std::thread prepareThread1(prepareData, context1, 1);
    std::thread processThread1(processData, context1);
    std::thread retrieveThread1(retrieveData, context1, 1);

    // Wait for some condition here, then set stop to true to stop the threads.
    stop = true;

    prepareThread0.join();
    processThread0.join();
    retrieveThread0.join();

    prepareThread1.join();
    processThread1.join();
    retrieveThread1.join();

    // checkError(cuModuleUnload(module0));
    checkError(cuCtxDestroy(context0));

    // checkError(cuModuleUnload(module1));
    checkError(cuCtxDestroy(context1));

    return 0;
}
