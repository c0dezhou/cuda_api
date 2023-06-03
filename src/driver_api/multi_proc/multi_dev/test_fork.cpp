#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#define SIZE_SMALL 10000
#define SIZE_MEDIUM 10000000
#define SIZE_LARGE 100000000

#define THREADS_PER_BLOCK 512

void run_calculation(int device, size_t size) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUdeviceptr d_data;
    float* h_data = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cuInit(0);
    cuDeviceGet(&cuDevice, device);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");

    cuModuleGetFunction(&cuFunction, cuModule, "_Z14vec_multiply_2Pfi");

    cuMemAlloc(&d_data, size * sizeof(float));

    cuMemcpyHtoD(d_data, h_data, size * sizeof(float));

    void* args[] = {&d_data, &size};
    cuLaunchKernel(cuFunction, size / THREADS_PER_BLOCK, 1, 1,
                   THREADS_PER_BLOCK, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();

    cuMemcpyDtoH(h_data, d_data, size * sizeof(float));

    cuMemFree(d_data);

    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

    for (int i = 0; i < size; i++) {
        ASSERT_EQ(h_data[i], i * 2.0f);
        // if (h_data[i] != i * 2) {
        //     printf("Error: data[%.2f] = %.2f\n", i, h_data[i]);
        // }
    }

    free(h_data);
}

TEST(MPROC, mdev_dev_fork_3dev) {
    GTEST_SKIP();
    int deviceCount;
    cuInit(0);
    cuDeviceGetCount(&deviceCount);
    if (deviceCount < 3) {
        printf("Error: requires at least 3 GPUs.\n");
        // return -1;
    }

    for (int i = 0; i < 3; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            size_t size;
            switch (i) {
                case 0:
                    size = SIZE_SMALL;
                    break;
                case 1:
                    size = SIZE_MEDIUM;
                    break;
                case 2:
                    size = SIZE_LARGE;
                    break;
            }
            run_calculation(i, size);
            // return 0;
        } else if (pid < 0) {
            printf("Error: fork failed.\n");
            // return -1;
        }
    }

    for (int i = 0; i < 3; i++) {
        wait(NULL);
    }
}
