#include "test_utils.h"

void not_increase_stream_perf(int d) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    CUmodule cuModule;
    checkError(cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction kernels[4];
    checkError(cuModuleGetFunction(&kernels[0], cuModule,
                                   "_Z18add_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&kernels[1], cuModule,
                                   "_Z18sub_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&kernels[2], cuModule,
                                   "_Z18mul_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&kernels[3], cuModule,
                                   "_Z18div_inplace_kernelPfS_i"));

    int N = 1024;
    size_t size = N * sizeof(float);

    CUdeviceptr d_a, d_b;
    checkError(cuMemAlloc(&d_a, size));
    checkError(cuMemAlloc(&d_b, size));

    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);

    checkError(cuMemcpyHtoD(d_a, h_a.data(), size));
    checkError(cuMemcpyHtoD(d_b, h_b.data(), size));

    // Single-Stream Execution
    CUstream cuStream;
    checkError(cuStreamCreate(&cuStream, CU_STREAM_NON_BLOCKING));

    CUevent start, stop;
    float elapsedTime;

    checkError(cuEventCreate(&start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    void *args[] = {&d_a, &d_b, &N};
    for (int i = 0; i < 4; i++) {
        checkError(cuEventRecord(start, cuStream));
        checkError(cuLaunchKernel(kernels[i], N/256 +1, 1, 1, 256, 1, 1, 0, cuStream, args, nullptr));
        checkError(cuEventRecord(stop, cuStream));
        checkError(cuEventSynchronize(stop));
        checkError(cuEventElapsedTime(&elapsedTime, start, stop));
        std::cout << "Single-stream execution time for operation " << i << ": " << elapsedTime << " ms" << std::endl;
    }
    checkError(cuStreamDestroy(cuStream));

    // Multi-Stream Execution
    const int numStreams = 32;
    CUstream cuStreams[numStreams];
    for(int i = 0; i < numStreams; i++){
        checkError(cuStreamCreate(&cuStreams[i], CU_STREAM_NON_BLOCKING));
    }

    for (int i = 0; i < numStreams; i++) {
        checkError(cuEventRecord(start, cuStreams[i]));
        checkError(cuLaunchKernel(kernels[i%4], N/256+1, 1, 1, 256, 1, 1, 0, cuStreams[i], args, nullptr));
        checkError(cuEventRecord(stop, cuStreams[i]));
        checkError(cuEventSynchronize(stop));
        checkError(cuEventElapsedTime(&elapsedTime, start, stop));
        std::cout << "Multi-stream execution time for operation " << i%4 << ": " << elapsedTime << " ms" << std::endl;
    }

    for(int i = 0; i < numStreams; i++){
        checkError(cuStreamDestroy(cuStreams[i]));
    }

    checkError(cuMemFree(d_a));
    checkError(cuMemFree(d_b));
    checkError(cuModuleUnload(cuModule));
    checkError(cuCtxDestroy(cuContext));

}

TEST(STRESS_NEW, not_increase_stream_perf) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        not_increase_stream_perf(d);
    }
}
