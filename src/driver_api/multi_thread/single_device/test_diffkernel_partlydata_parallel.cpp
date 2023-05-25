#include "test_utils.h"

void cudaThread(int threadId,
                CUcontext context,
                CUdeviceptr d_data,
                int dataSize,
                int startIndex,
                int endIndex,
                CUfunction kernel) {
    checkError(cuCtxSetCurrent(context));
    CUstream stream;
    cuStreamCreate(&stream, 0);
    int blockSize = 256;
    int gridSize = (endIndex - startIndex + blockSize - 1) / blockSize;

    void* kernelArgs[] = {&d_data, &startIndex, &endIndex};
    cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream,
                   kernelArgs, nullptr);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);
}

TEST(MthsTest_, MTH_Single_Device_diffkernel_partlydata_parallel) {
    int dataSize = 10;
    int numThreads = 3;
    int dataSizePerThread = dataSize / numThreads;

    checkError(cuInit(0));
    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    CUcontext context;
    checkError(cuCtxCreate(&context, 0, device));

    std::vector<float> h_data(dataSize, 1.0f);

    CUdeviceptr d_data;
    cuMemAlloc(&d_data, dataSize * sizeof(float));

    cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float));

    CUmodule module;
    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction vec_multiply_2, vec_add_3, vec_sub_1;
    checkError(cuModuleGetFunction(&vec_multiply_2, module,
                                   "_Z22vec_multiply_2_withidxPfii"));
    checkError(
        cuModuleGetFunction(&vec_add_3, module, "_Z17vec_add_3_withidxPfii"));

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

    for (auto& thread : threads) {
        thread.join();
    }

    cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float));

    bool result = verifyResult(h_data, 2.0f, 0, dataSize / 3);
    std::cout << "Kernel 1 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(h_data, 4.0f, dataSize / 3, (dataSize / 3) * 2);
    std::cout << "Kernel 2 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(h_data, 2.0f, (dataSize / 3) * 2, dataSize);
    std::cout << "Kernel 1 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    checkError(cuMemFree(d_data));
    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));

}
