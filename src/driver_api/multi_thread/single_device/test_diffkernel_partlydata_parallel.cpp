#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

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

bool verifyResult(const std::vector<float>& data,
                  float expectedValue,
                  int startIndex,
                  int endIndex) {
    for (int i = startIndex; i < endIndex; i++) {
        if (data[i] != expectedValue) {
            return false;
        }
    }
    return true;
}

// CUDA thread function
void cudaThread(int threadId,
                CUcontext context,
                CUdeviceptr d_data,
                int dataSize,
                int startIndex,
                int endIndex,
                CUfunction kernel) {
    CHECK_CUDA(cuCtxSetCurrent(context));
    // Create CUDA stream
    CUstream stream;
    cuStreamCreate(&stream, 0);

    // Calculate the grid and block dimensions
    int blockSize = 256;
    int gridSize = (endIndex - startIndex + blockSize - 1) / blockSize;

    // Launch the kernel based on the threadId
    void* kernelArgs[] = {&d_data, &startIndex, &endIndex};
    cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream,
                   kernelArgs, nullptr);

    // Synchronize the stream
    cuStreamSynchronize(stream);

    // Destroy the stream
    cuStreamDestroy(stream);
}

int main() {
    int dataSize = 10;                         // Size of the data
    int numThreads = 3;                            // Number of threads
    int dataSizePerThread = dataSize / numThreads;  // Data size per thread

    CHECK_CUDA(cuInit(0));
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Allocate host memory for data
    std::vector<float> h_data(dataSize, 1.0f);

    // Allocate device memory for data
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, dataSize * sizeof(float));

    // Copy data from host to device
    cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float));

    CUmodule module;
    CHECK_CUDA(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction vec_multiply_2, vec_add_3, vec_sub_1;
    CHECK_CUDA(cuModuleGetFunction(&vec_multiply_2, module,
                                   "_Z22vec_multiply_2_withidxPfii"));
    CHECK_CUDA(
        cuModuleGetFunction(&vec_add_3, module, "_Z17vec_add_3_withidxPfii"));

    // Create threads
    std::vector<std::thread> threads(numThreads);
    for (int threadId = 0; threadId < numThreads; ++threadId) {
        int startIndex = threadId * dataSizePerThread;
        int endIndex = startIndex + dataSizePerThread;
        if (threadId == numThreads - 1) {
            endIndex += dataSize % numThreads;
        }
        CUfunction cuKernel;
        if (threadId % 2 == 0) {
            cuKernel = vec_multiply_2;
        } else {
            cuKernel = vec_add_3;
        }
        threads[threadId] =
            std::thread(cudaThread, threadId, context, d_data, dataSize,
                        startIndex, endIndex, cuKernel);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Copy data from device to host
    cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float));

    bool result =
        verifyResult(h_data, 2.0f, 0, dataSize / 3);  // Verify kernel1 result
    std::cout << "Kernel 1 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(h_data, 4.0f, dataSize / 3,
                          (dataSize / 3) * 2);  // Verify kernel2 result
    std::cout << "Kernel 2 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(h_data, 2.0f, (dataSize / 3)*2,
                          dataSize);  // Verify kernel2 result
    std::cout << "Kernel 2 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    // Free device memory

    CHECK_CUDA(cuMemFree(d_data));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}
