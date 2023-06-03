#include <cuda.h>
#include <thread>
#include <stdio.h>
#include <gtest/gtest.h>

#define SIZE_LARGE 1000000000
#define THREADS_PER_BLOCK 512
#define ROUNDS 10
#define DEVICE_COUNT 8

struct ThreadData {
    int device;
};

void run_calculation5(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    int device = data->device;

    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    cuInit(0);
    cuDeviceGet(&cuDevice, device);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");

    cuModuleGetFunction(&cuFunction, cuModule, "_Z14vec_multiply_2Pfi");

    for (int round = 0; round < ROUNDS; round++) {
        CUdeviceptr d_data;
        float* h_data = (float*)malloc(SIZE_LARGE * sizeof(float));

        for (int i = 0; i < SIZE_LARGE; i++) {
            h_data[i] = i;
        }

        cuMemAlloc(&d_data, SIZE_LARGE * sizeof(float));

        cuMemcpyHtoD(d_data, h_data, SIZE_LARGE * sizeof(float));

        int n = SIZE_LARGE;
        void* args[] = {&d_data, &n};
        cuLaunchKernel(cuFunction, SIZE_LARGE / THREADS_PER_BLOCK, 1, 1,
                       THREADS_PER_BLOCK, 1, 1, 0, NULL, args, NULL);
        cuCtxSynchronize();

        cuMemcpyDtoH(h_data, d_data, SIZE_LARGE * sizeof(float));

        cuMemFree(d_data);

        for (int i = 0; i < SIZE_LARGE; i++) {
            ASSERT_EQ(h_data[i], i * 2.0f);
            // if (h_data[i] != i * 2) {
            //     printf("Error: data[%.2f] = %.2f\n", i, h_data[i]);
            // }
        }

        free(h_data);
    }

    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

}

TEST(MTHS, muldev_largdata) {
    struct ThreadData threadData[DEVICE_COUNT];

    std::thread threads[DEVICE_COUNT];

    for (int i = 0; i < DEVICE_COUNT; i++) {
        threadData[i].device = i;
        threads[i] = std::thread(run_calculation5, &threadData[i]);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
