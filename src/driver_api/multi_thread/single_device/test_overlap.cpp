// 将主机和设备之间的数据传输与内核执行重叠
#include "test_utils.h"

#define NUM_THREADS 2
#define N 1024

void threadFunc(int threadId,
                float* hostA,
                float* hostB,
                float* hostC,
                int size) {
    CUcontext context;
    CUdevice device;
    CUmodule module;
    CUfunction kernelFunc;
    CUstream stream;
    CUdeviceptr deviceA, deviceB, deviceC;

    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
    checkError(
        cuModuleGetFunction(&kernelFunc, module, "_Z10add_kernelPfS_S_i"));

    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    checkError(cuMemAlloc(&deviceA, size * sizeof(float)));
    checkError(cuMemcpyHtoDAsync(deviceA, hostA, size * sizeof(float), stream));

    checkError(cuMemAlloc(&deviceB, size * sizeof(float)));
    checkError(cuMemcpyHtoDAsync(deviceB, hostB, size * sizeof(float), stream));

    checkError(cuMemAlloc(&deviceC, size * sizeof(float)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void* args[] = {&deviceA, &deviceB, &deviceC, &size};
    checkError(cuLaunchKernel(kernelFunc, gridSize, 1, 1, blockSize, 1, 1, 0,
                              stream, args, nullptr));

    checkError(cuMemcpyDtoHAsync(hostC, deviceC, size * sizeof(float), stream));
    checkError(cuStreamSynchronize(stream));

    checkError(cuStreamDestroy(stream));
    checkError(cuMemFree(deviceA));
    checkError(cuMemFree(deviceB));
    checkError(cuMemFree(deviceC));
    checkError(cuCtxDestroy(context));
}

TEST(MthsTest_, MTH_Single_Device_overlap) {
    checkError(cuInit(0));

    float *hostA, *hostB, *hostC;
    hostA = new float[N];
    hostB = new float[N];
    hostC = new float[N];

    for (int i = 0; i < N; ++i) {
        hostA[i] = i;
        hostB[i] = i;
    }

    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i] = std::thread(threadFunc, i, hostA, hostB, hostC, N);
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < N; ++i) {
        if (hostC[i] != 2 * i) {
            std::cerr << "Check failed at index " << i << "! Expected " << 2 * i
                      << ", got " << hostC[i] << std::endl;
            // return EXIT_FAILURE;
            exit(1);
        }
    }
    std::cout << "Check passed!" << std::endl;

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;

}
