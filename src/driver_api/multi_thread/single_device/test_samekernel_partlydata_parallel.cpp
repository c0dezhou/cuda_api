#include "test_utils.h"

void cudaOperation(int threadId,
                   float* h_C,
                   CUdeviceptr d_A,
                   CUdeviceptr d_B,
                   CUdeviceptr d_C,
                   int size,
                   int numThreads) {

    cuCtxSetCurrent(NULL);

    CUcontext context;
    cuCtxCreate(&context, 0, 0);

    int dataSizePerThread = size / numThreads;
    int startIndex = threadId * dataSizePerThread;
    int endIndex = startIndex + dataSizePerThread;
    if (threadId == numThreads - 1) {
        endIndex += size % numThreads;
    }

    CUresult result;
    result = cuMemAlloc(&d_A, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    for (int i = 0; i < size; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    result =
        cuMemcpyHtoD(d_A + startIndex * sizeof(float), h_A.data() + startIndex,
                     (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result =
        cuMemcpyHtoD(d_B + startIndex * sizeof(float), h_B.data() + startIndex,
                     (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);

    CUmodule module;
    result = cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    assert(result == CUDA_SUCCESS);

    CUfunction function;
    result = cuModuleGetFunction(&function, module, "_Z10add_kernelPfS_S_i");
    assert(result == CUDA_SUCCESS);

    int blockSize = 256;
    int gridSize = ((endIndex - startIndex) + blockSize - 1) / blockSize;
    CUdeviceptr new_d_A = d_A + startIndex * sizeof(float);
    CUdeviceptr new_d_B = d_B + startIndex * sizeof(float);
    CUdeviceptr new_d_C = d_C + startIndex * sizeof(float);

    void* kernelParams[] = {&new_d_A, &new_d_B, &new_d_C, &size};
    result = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, NULL,
                            kernelParams, NULL);
    assert(result == CUDA_SUCCESS);

    result = cuMemcpyDtoH(h_C + startIndex, new_d_C,
                          (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);

    result = cuMemFree(d_A);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_B);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_C);
    assert(result == CUDA_SUCCESS);

    cuCtxDestroy(context);
}

TEST(MthsTest_, MTH_Single_Device_samekernel_partlydata_parallel) {
    cuInit(0);

    const int numThreads = 20;
    const int dataSize = 1000;

    std::vector<std::thread> threads(numThreads);
    std::vector<CUdeviceptr> d_A(numThreads);
    std::vector<CUdeviceptr> d_B(numThreads);
    std::vector<CUdeviceptr> d_C(numThreads);

    std::vector<float> h_C(dataSize);

    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(cudaOperation, i, h_C.data(), d_A[i], d_B[i], d_C[i],
                                 dataSize, numThreads);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "All threads finished executing CUDA operations." << std::endl;

    for (int i = 0; i < dataSize; ++i) {
        float expected = i + i;
        float actual = h_C[i];
        if (actual != expected) {
            std::cout << "Verification failed at index " << i << ": expected "
                      << expected << ", actual " << actual << std::endl;
            exit(1);
        }
    }

    std::cout << "Verification succeeded. The final result is reasonable."
              << std::endl;

}
