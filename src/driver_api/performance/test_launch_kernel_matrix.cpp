#include "test_utils.h"


std::string loadPtx(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::string result(std::istreambuf_iterator<char>{file}, {});
    return result;
}

void benchmarkKernel(CUfunction function,
                     int blocks_x,
                     int blocks_y,
                     int threads_x,
                     int threads_y,
                     void** args,
                     int numLaunches,
                     float maxDeviation) {
    std::vector<float> times(numLaunches);

    CUevent start, stop;
    checkError(cuEventCreate(&start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    for (int i = 0; i < numLaunches; i++) {
        checkError(cuEventRecord(start, 0));
        checkError(cuLaunchKernel(function, blocks_x, blocks_y, 1, threads_x,
                                  threads_y, 1, 0, 0, args, 0));
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

TEST(PERF, launch_matrix_kernel_loop_multidevice) {
    checkError(cuInit(0));

    int deviceCount;
    checkError(cuDeviceGetCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Requires at least two GPUs" << std::endl;
        // return -1;
    }

    std::string ptx = loadPtx(
        "/data/system/yunfan/cuda_api/common/cuda_kernel/stress_kernel.ptx");

    int M = 1000, N = 1000, K = 1000;
    float *h_A, *h_B, *h_C;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = distribution(generator);
        h_B[i] = distribution(generator);
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;

    checkError(cuMemAlloc((CUdeviceptr*)&d_A, M * K * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_B, K * N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_C, M * N * sizeof(float)));

    checkError(cuMemcpyHtoD((CUdeviceptr)d_A, h_A, M * K * sizeof(float)));
    checkError(cuMemcpyHtoD((CUdeviceptr)d_B, h_B, K * N * sizeof(float)));

    void *args[] = {d_A, d_B, d_C, &M, &N, &K};

    int threads_x = 32;
    int threads_y = 32;
    int blocks_x = (N + threads_x - 1) / threads_x;
    int blocks_y = (M + threads_y - 1) / threads_y;


    std::vector<float> averageTimes(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        CUdevice device;
        checkError(cuDeviceGet(&device, deviceId));

        CUcontext context;
        checkError(cuCtxCreate(&context, 0, device));

        CUmodule module;
        checkError(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0));

        CUfunction function;
        checkError(cuModuleGetFunction(&function, module,
                                       "_Z20matrixMultiplyKernelPKfS0_Pfiii"));

        benchmarkKernel(function, blocks_x, blocks_y, threads_x, threads_y, args, 100, 1.1);

        checkError(cuModuleUnload(module));
        checkError(cuCtxDestroy(context));
    }

    checkError(cuMemFree((CUdeviceptr)d_A));
    checkError(cuMemFree((CUdeviceptr)d_B));
    checkError(cuMemFree((CUdeviceptr)d_C));

    // Free host memory...
    // ...

}
