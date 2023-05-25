// 每个设备启动一个单独的线程。 每个线程处理其相应设备上的一部分数据

#include "test_utils.h"

void process_samekernel(int deviceIndex, int portionSize, float* hostData) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr deviceData;

    cuDeviceGet(&device, deviceIndex);
    cuCtxCreate(&context, 0, device);
    checkError(cuModuleLoad(&module,
                            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                            "cuda_kernel.ptx"));
    checkError(cuModuleGetFunction(&kernel, module, "_Z9vec_sub_1Pfi"));
    cuMemAlloc(&deviceData, portionSize * sizeof(float));
    cuMemcpyHtoD(deviceData, hostData, portionSize * sizeof(float));

    void* args[] = {&deviceData, &portionSize};
    cuLaunchKernel(kernel, (portionSize + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);

    cuCtxSynchronize();
    cuMemcpyDtoH(hostData, deviceData, portionSize * sizeof(float));

    cuMemFree(deviceData);
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

TEST(MthsTest_, MTH_multi_Device_per_th_per_dev_samekernel) {
    checkError(cuInit(0));

    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);

    const int dataSize = 16;
    const int portionSize = dataSize / deviceCount;
    float* hostData = new float[portionSize * deviceCount]();

    std::vector<std::thread> threads(deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        // threads[i] = std::thread(process_samekernel, i, portionSize, hostData + i * portionSize);
        threads[i] = std::thread(process_samekernel, i, portionSize * deviceCount,
                                 hostData);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int i = 0; i < dataSize; i++) {
        if (hostData[i] != -2.0f) {
            std::cout << "Result verification failed at element " << i << ". Expected 0, got " << hostData[i] << std::endl;
            delete[] hostData;
            exit(1);
        }
    }

    std::cout << "Result verification succeeded!" << std::endl;
    delete[] hostData;
}
