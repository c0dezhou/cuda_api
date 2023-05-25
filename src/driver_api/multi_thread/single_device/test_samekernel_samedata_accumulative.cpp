#include "test_utils.h"

void cudaOperation(int threadId,
                   CUdeviceptr d_A,
                   CUdeviceptr d_B,
                   CUdeviceptr d_C,
                   int size) {
    cuCtxSetCurrent(NULL);

    CUcontext context;
    cuCtxCreate(&context, 0, 0);

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
        h_A[i] = i + threadId;
        h_B[i] = i - threadId;
    }

    result = cuMemcpyHtoD(d_A, h_A.data(), size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemcpyHtoD(d_B, h_B.data(), size * sizeof(float));
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
    int gridSize = (size + blockSize - 1) / blockSize;
    void* kernelParams[] = {&d_A, &d_B, &d_C, &size};
    result = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, NULL,
                            kernelParams, NULL);
    assert(result == CUDA_SUCCESS);

    std::vector<float> h_C(size);
    result = cuMemcpyDtoH(h_C.data(), d_C, size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    bool res = verifyResult(h_A, h_B, h_C);
    assert(res);

    result = cuMemFree(d_A);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_B);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_C);
    assert(result == CUDA_SUCCESS);

    cuCtxDestroy(context);
}

TEST(MthsTest_, MTH_Single_Device_samekernel_samedata_accumulative) {
    cuInit(0);

    const int numThreads = 20;
    const int dataSize = 1000;

    std::vector<std::thread> threads(numThreads);
    std::vector<CUdeviceptr> d_A(numThreads);
    std::vector<CUdeviceptr> d_B(numThreads);
    std::vector<CUdeviceptr> d_C(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        threads[i] =
            std::thread(cudaOperation, i, d_A[i], d_B[i], d_C[i], dataSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "All threads finished executing CUDA operations." << std::endl;

}
