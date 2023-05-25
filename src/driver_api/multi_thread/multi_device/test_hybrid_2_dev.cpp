#include "test_utils.h"

#define dataSize 1024


std::vector<float> data(dataSize);
std::vector<float> results(dataSize);



void processOnDevice(CUdevice device, int deviceIndex) {
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr devData;
    CUdeviceptr devResults;
    int offset = deviceIndex * dataSize / 2;

    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuModuleLoad(&module,
                            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                            "cuda_kernel.ptx"));
    checkError(cuModuleGetFunction(&kernel, module, "_Z14vec_multiply_2Pfi"));

    checkError(cuMemAlloc(&devData, dataSize * sizeof(float) / 2));
    // checkError(cuMemAlloc(&devResults, dataSize * sizeof(float) / 2));

    checkError(cuMemcpyHtoD(devData, &data[offset], dataSize * sizeof(float) / 2));

    auto ture_size = dataSize / 2;
    // void* args[] = {&devData, &devResults, &ture_size};
    void* args[] = {&devData, &ture_size};
    checkError(cuLaunchKernel(kernel, dataSize / 2, 1, 1, 1, 1, 1, 0, 0, args, 0));

    checkError(
        cuMemcpyDtoH(&results[offset], devData, dataSize * sizeof(float) / 2));

    checkError(cuMemFree(devData));
    checkError(cuMemFree(devResults));
    checkError(cuCtxDestroy(context));
}

TEST(MthsTest_, MTH_multi_Device_hybrid_cpu_gpu) {
    checkError(cuInit(0));

    CUdevice device0, device1;
    checkError(cuDeviceGet(&device0, 0));
    checkError(cuDeviceGet(&device1, 1));

    // Initialize data
    for (int i = 0; i < dataSize; i++) {
        data[i] = static_cast<float>(i);
    }

    std::thread cpuThread(cpuComputation, std::ref(data));

    std::thread processThread0(processOnDevice, device0, 0);
    std::thread processThread1(processOnDevice, device1, 1);

    cpuThread.join();
    processThread0.join();
    processThread1.join();

    // check results
    for (int i = 0; i < dataSize; i++) {
        std::cout << results[i] << std::endl;
        if (results[i] != 4.0f * i) {
            std::cout << "Error: result mismatch at position " << i << std::endl;
            exit(1);
        }
    }

    std::cout << "All results are correct." << std::endl;

}
