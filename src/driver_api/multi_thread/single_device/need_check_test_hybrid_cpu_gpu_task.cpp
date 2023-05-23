
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>

#define NUM_THREADS 4

void threadFunc(int threadId, float* data, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module1, module2;
    CUfunction kernelFunc1, kernelFunc2;
    CUstream stream;
    CUevent startEvent1, endEvent1, startEvent2, endEvent2;
    CUresult cuResult;

    cuResult = cuDeviceGet(&device, 0);
    cuResult = cuCtxCreate(&context, 0, device);

    cuResult = cuModuleLoad(&module1,
                            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                            "cuda_kernel.ptx");
    // cuResult = cuModuleLoad(&module2, "cuda_kernels2.ptx"); //c处理两个load
    cuResult =
        cuModuleGetFunction(&kernelFunc1, module1, "_Z9vec_sub_1Pfi");
    cuResult = cuModuleGetFunction(&kernelFunc2, module2, "_Z9vec_sub_1Pfi");

    cuResult = cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    cuResult = cuEventCreate(&startEvent1, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&endEvent1, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&startEvent2, CU_EVENT_DEFAULT);
    cuResult = cuEventCreate(&endEvent2, CU_EVENT_DEFAULT);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    if (threadId > 0) {
        cuResult = cuEventSynchronize(endEvent2);
    }

    cuResult = cuEventRecord(startEvent1, stream);

    void * args1[] = {&data, &size};
    cuResult = cuLaunchKernel(kernelFunc1, gridSize, 1, 1, blockSize, 1, 1, 0,
                              stream, args1, nullptr);

    cuResult = cuEventRecord(endEvent1, stream);

    if (threadId < NUM_THREADS - 1) {
        cuResult = cuEventSynchronize(startEvent2);
    }

    cuResult = cuEventRecord(startEvent2, stream);

    cuResult = cuLaunchKernel(kernelFunc2, gridSize, 1, 1, blockSize, 1, 1, 0,
                              stream, args1, nullptr);

    cuResult = cuEventRecord(endEvent2, stream);

    cuResult = cuStreamSynchronize(stream);

    cuResult = cuEventDestroy(startEvent1);
    cuResult = cuEventDestroy(endEvent1);
    cuResult = cuEventDestroy(startEvent2);
    cuResult = cuEventDestroy(endEvent2);

    cuResult = cuStreamDestroy(stream);

    cuResult = cuCtxDestroy(context);
}

int main() {
    cuInit(0);
    int dataSize = 1024;
    std::vector<float> data(dataSize);

    for (int i = 0; i < dataSize; i++) {
        data[i] = static_cast<float>(i);
    }

    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(threadFunc, i, data.data(), dataSize);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    bool isCorrect = true;
    for (int i = 0; i < dataSize; i++) {
        if (data[i] != (2.0f * i + 1.0f)) {
            isCorrect = false;
            break;
        }
    }

    if (isCorrect) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is incorrect!" << std::endl;
    }

    return 0;
}
