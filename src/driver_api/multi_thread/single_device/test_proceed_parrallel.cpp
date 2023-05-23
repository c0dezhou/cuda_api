#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>

#define NUM_THREADS 4
#define DATA_SIZE 1024

void processThread(CUdeviceptr d_data,
                   float* data,
                   int startIndex,
                   int endIndex,
                   CUcontext context,
                   CUfunction kernel) {
    CUresult cuResult;
    cuCtxSetCurrent(context);

    void* args[] = {&d_data, &startIndex, &endIndex};

    int blockSize = 256;
    int gridSize = (DATA_SIZE + blockSize - 1) / blockSize;

    cuResult = cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, NULL,
                              args, nullptr);

    cuResult = cuCtxSynchronize();

    cuResult = cuMemcpyDtoH(data + startIndex, d_data + startIndex,
                            (endIndex - startIndex) * sizeof(float));
}

int main() {
    CUresult cuResult;
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernel;

    cuInit(0);
    std::vector<float> data(DATA_SIZE, 1.0f);

    cuResult = cuDeviceGet(&device, 0);
    cuResult = cuCtxCreate(&context, 0, device);

    cuResult = cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    cuResult =
        cuModuleGetFunction(&kernel, module, "_Z22vec_multiply_2_withidxPfii");

    CUdeviceptr d_data;
    cuResult = cuMemAlloc(&d_data, DATA_SIZE * sizeof(float));
    cuResult = cuMemcpyHtoD(d_data, data.data(), DATA_SIZE * sizeof(float));

    std::vector<std::thread> threads(NUM_THREADS);
    int dataSizePerThread = DATA_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int startIndex = i * dataSizePerThread;
        int endIndex = startIndex + dataSizePerThread;
        if (i == NUM_THREADS - 1) {
            endIndex += DATA_SIZE % NUM_THREADS;
        }

        threads[i] = std::thread(processThread, d_data, data.data(), startIndex,
                                 endIndex, context, kernel);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int i = 0; i < DATA_SIZE; ++i) {
        if (data[i] != 2.0f) {
            std::cout << "Incorrect result at index " << i << "num is "
            << data[i] <<std::endl;
            break;
        }
    }

    std::cout << "Result verification complete." << std::endl;
    
    cuResult = cuMemFree(d_data);
    cuResult = cuModuleUnload(module);
    cuResult = cuCtxDestroy(context);
    return 0;
}
