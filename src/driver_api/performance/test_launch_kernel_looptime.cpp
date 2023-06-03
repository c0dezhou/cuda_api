#include "test_utils.h"

void benchmarkKernel(CUfunction function, int blocks, int threads, void** args, int numLaunches, float maxDeviation) {
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


std::string loadptx(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::string result(std::istreambuf_iterator<char>{file}, {});
    return result;
}

TEST(PERF, launch_empty_kernel_loop_onedevice) {
    checkError(cuInit(0));

    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    CUcontext context;
    checkError(cuCtxCreate(&context, 0, device));

    CUmodule module;
    std::string ptx = loadptx(
        "/data/system/yunfan/cuda_api/common/cuda_kernel/"
        "resource_limitation_kernel.ptx");
    checkError(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0));

    CUfunction function;
    checkError(cuModuleGetFunction(&function, module, "_Z12dummy_kernelv"));

    int loops = 1000;
    // void *args[] = {&loops};
    void *args[] = {};

    int threads = 256;
    int blocks = (loops + threads - 1) / threads;

    benchmarkKernel(function, blocks, threads, args, 100, 1.1);

    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));

}
