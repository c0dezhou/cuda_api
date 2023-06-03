#include "test_utils.h"


void benchmarkKernel1(CUfunction function, int blocks, int threads, void** args, int numLaunches, float maxDeviation) {
    std::vector<float> times(numLaunches);

    CUevent start, stop;
    checkError(cuEventCreate(&start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    for (int i = 0; i < numLaunches; i++) {
        checkError(cuEventRecord(start, 0));
        checkError(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
        checkError(cuEventRecord(stop, 0));
        checkError(cuCtxSynchronize());

        float milliseconds = 0;
        checkError(cuEventElapsedTime(&milliseconds, start, stop));
        times[i] = milliseconds;
    }

    float averageTime = std::accumulate(times.begin(), times.end(), 0.0f) / numLaunches;
    float maxTime = averageTime * maxDeviation;

    for (int i = 0; i < numLaunches; i++) {
        if (times[i] > maxTime) {
            std::cout << "Kernel execution time for launch " << i << " exceeded " << maxDeviation << " times the average." << std::endl;
        }
    }
}

TEST(PERF,launch_empty_kernel_loop_multidevice) {
    checkError(cuInit(0));

    int deviceCount;
    checkError(cuDeviceGetCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Requires at least two GPUs" << std::endl;
        // return -1;
    }

    // std::string ptx = loadPtx("emptyKernel.ptx");

    int loops = 1000;
    // void *args[] = {&loops};
    void* args[] = {};

    int threads = 256;
    int blocks = (loops + threads - 1) / threads;

    std::vector<float> averageTimes(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        CUdevice device;
        checkError(cuDeviceGet(&device, deviceId));

        CUcontext context;
        checkError(cuCtxCreate(&context, 0, device));

        CUmodule module;
        // checkError(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0));
        checkError(
            cuModuleLoad(&module,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "resource_limitation_kernel.ptx"));

            CUfunction function;
            checkError(
                cuModuleGetFunction(&function, module, "_Z12dummy_kernelv"));

            benchmarkKernel1(function, blocks, threads, args, 100, 1.1);

            checkError(cuModuleUnload(module));
            checkError(cuCtxDestroy(context));
        }

    float minTime = *std::min_element(averageTimes.begin(), averageTimes.end());
    float maxTime = *std::max_element(averageTimes.begin(), averageTimes.end());

    if(maxTime > minTime * 1.1){
        std::cout << "The gap between devices is too large!" << std::endl;
    }

    // return 0;
}
