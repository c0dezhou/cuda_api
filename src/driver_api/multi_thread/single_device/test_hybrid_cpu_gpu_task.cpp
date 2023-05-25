#include "test_utils.h"

#define NUM_THREADS 4


void threadFunc(int threadId, std::vector<float>& data, int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;

    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
    checkError(cuModuleGetFunction(&kernelFunc, module, "_Z9vec_sub_1Pfi"));

    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    cpuComputation(data);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    CUdeviceptr d_data;
    size_t bytes = size * sizeof(float);
    checkError(cuMemAlloc(&d_data, bytes));
    checkError(cuMemcpyHtoD(d_data, data.data(), bytes));

    void* args[] = {&d_data, &size};
    checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0,
                              stream, args, nullptr));
    checkError(cuStreamSynchronize(stream));

    checkError(cuMemcpyDtoH(data.data(), d_data, bytes));
    checkError(cuMemFree(d_data));

    checkError(cuStreamDestroy(stream));
    checkError(cuCtxDestroy(context));
}

TEST(MthsTest_, MTH_Single_Device_hybrid_cpu_gpu) {
    checkError(cuInit(0));
    int dataSize = 10;
    std::vector<float> data(dataSize, 1.0f);

    // 1 * 2 - 1 没加锁
    // 17 = ((((2*2-1)*2-1)*2 -1)*2 -1)
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] =
            std::thread(threadFunc, i, std::ref(data),
                        dataSize);  // std::ref 通常用于传递对可复制对象的引用
    }

    for (auto& t : threads) {
        t.join();
    }

    for (const auto& value : data) {
        std::cout << value << std::endl;
    }
}
