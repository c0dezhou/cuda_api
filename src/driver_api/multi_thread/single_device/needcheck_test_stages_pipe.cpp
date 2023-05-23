// 并发执行不同数据处理阶段

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>

// g++ -o test_stages_pipe test_stages_pipe.cpp
// -I/usr/local/cuda-11.3/targets/x86_64-linux/include -lcuda -lpthread

#define NUM_THREADS 4

#define CHECK_CUDA(op) __check_cuda_error((op), #op, __FILE__, __LINE__)

bool __check_cuda_error(CUresult code,
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

void threadFunc(int threadId, float* data, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUstream stream;
    CUfunction vec_multiply_2, vec_add_3;
    CUevent startEvent1, endEvent1, startEvent2, endEvent2;
    CUresult cuResult;

    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    CHECK_CUDA(cuCtxSetCurrent(context));
    CHECK_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CHECK_CUDA(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CHECK_CUDA(
        cuModuleGetFunction(&vec_multiply_2, module, "_Z14vec_multiply_2Pfi"));
    CHECK_CUDA(cuModuleGetFunction(&vec_add_3, module, "_Z9vec_add_3Pfi"));

    CHECK_CUDA(cuEventCreate(&startEvent1, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&endEvent1, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&startEvent2, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&endEvent2, CU_EVENT_DEFAULT));

    void* args1[] = {&data, &size};
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    if (threadId % 2 == 0) {
        CHECK_CUDA(cuEventRecord(startEvent1, stream));
        CHECK_CUDA(cuLaunchKernel(vec_multiply_2, gridSize, 1, 1, blockSize, 1,
                                  1, 0, stream, args1, nullptr));
        CHECK_CUDA(cuEventRecord(endEvent1, stream));
        CHECK_CUDA(cuEventSynchronize(endEvent1));
    } else {
        CHECK_CUDA(cuEventSynchronize(endEvent1));
        CHECK_CUDA(cuEventRecord(startEvent2, stream));
        CHECK_CUDA(cuLaunchKernel(vec_add_3, gridSize, 1, 1, blockSize, 1, 1, 0,
                                  stream, args1, nullptr));
        CHECK_CUDA(cuEventRecord(endEvent2, stream));
        CHECK_CUDA(cuEventSynchronize(endEvent2));
    }

    CHECK_CUDA(cuEventDestroy(startEvent1));
    CHECK_CUDA(cuEventDestroy(endEvent1));
    CHECK_CUDA(cuEventDestroy(startEvent2));
    CHECK_CUDA(cuEventDestroy(endEvent2));
    CHECK_CUDA(cuStreamDestroy(stream));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));
}

int main() {
    cuInit(0);

    int dataSize = 1000;
    std::vector<float> data(dataSize, 1.0f);

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(threadFunc, i, data.data(), dataSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify the result
    bool isCorrect = true;
    for (int i = 0; i < dataSize; ++i) {
        if (i % 2 == 0) {
            if (data[i] != 2.0f) {
                isCorrect = false;
                break;
            }
        } else {
            if (data[i] != 4.0f) {
                isCorrect = false;
                break;
            }
        }
    }

    // Print result
    if (isCorrect) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is incorrect!" << std::endl;
    }

    return 0;
}
