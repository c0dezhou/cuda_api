#include "test_utils.h"

#define NUM_THREADS 4

void threadFunc(int threadId, float* hostData, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;
    CUdeviceptr deviceData;

    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
    if (threadId == 0)
        checkError(
            cuModuleGetFunction(&kernelFunc, module, "_Z14vec_multiply_2Pfi"));
    else
        checkError(cuModuleGetFunction(&kernelFunc, module, "_Z9vec_add_3Pfi"));

    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    checkError(cuMemAlloc(&deviceData, size * sizeof(float)));
    checkError(cuMemcpyHtoD(deviceData, hostData, size * sizeof(float)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void* args[] = {&deviceData, &size};
    checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0,
                              stream, args, nullptr));
    checkError(cuStreamSynchronize(stream));

    checkError(cuMemcpyDtoH(hostData, deviceData, size * sizeof(float)));

    checkError(cuStreamDestroy(stream));
    checkError(cuMemFree(deviceData));
    checkError(cuCtxDestroy(context));
}

TEST(MthsTest_, MTH_Single_Device_concurrent_kernel) {
    checkError(cuInit(0));
    int dataSize = 1024 * NUM_THREADS;
    std::vector<float> data(dataSize, 0.0f);

    std::vector<std::thread> threads(NUM_THREADS);
    int chunkSize = dataSize / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] =
            std::thread(threadFunc, i, data.data() + i * chunkSize, chunkSize);
    }

    for (auto& t : threads) {
        t.join();
    }

    // for (const auto &value : data) {
    //     std::cout << value << std::endl;
    // }

    bool result = verifyResult(data, 0.0f, 0, chunkSize);
    std::cout << "Kernel 1 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

    result = verifyResult(data, 3.0f, chunkSize, dataSize);
    std::cout << "Kernel 2 result verification: "
              << (result ? "Passed" : "Failed") << std::endl;

}
