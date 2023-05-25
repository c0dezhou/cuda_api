#include "test_utils.h"


void process_diffkernel(int deviceIndex, int portionSize, float* hostData) {
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
    if(deviceIndex == 0)
        checkError(cuModuleGetFunction(&kernel, module, "_Z9vec_sub_1Pfi"));
    else
        checkError(cuModuleGetFunction(&kernel, module, "_Z9vec_add_3Pfi"));

    cuMemAlloc(&deviceData, portionSize * sizeof(float));
    cuMemcpyHtoD(deviceData, hostData, portionSize * sizeof(float)); // Copy host data to device

    void* args[] = {&deviceData, &portionSize};
    cuLaunchKernel(kernel, (portionSize + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    

    cuCtxSynchronize();
    cuMemcpyDtoH(hostData, deviceData, portionSize * sizeof(float)); // Copy result back to host

    cuMemFree(deviceData);
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

TEST(MthsTest_, MTH_multi_Device_per_th_per_dev_diffkernel) {
    checkError(cuInit(0));

    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);

    const int dataSize = 1e6;
    const int portionSize = dataSize / deviceCount;
    float* hostData = new float[portionSize * deviceCount]();
    // float hostData[1000000] = {1};

    std::vector<std::thread> threads(deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        threads[i] = std::thread(process_diffkernel, i, portionSize, hostData + i * portionSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int i = 0; i < dataSize; i++) {
        float expected = (i < dataSize / 2) ? -1.0f : 3.0f;
        if (hostData[i] != expected) {
            std::cout << "Result verification failed at element " << i << ". Expected " << expected << ", got " << hostData[i] << std::endl;
            delete[] hostData;
            exit(1);
        }
    }

    std::cout << "Result verification succeeded!" << std::endl;
    delete[] hostData;
}
