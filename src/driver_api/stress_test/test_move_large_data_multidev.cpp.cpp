#include "test_utils.h"

#define SIZE_LARGE 1000000000
#define THREADS_PER_BLOCK 512
#define ROUNDS 10
#define DEVICE_COUNT 2

void run_calculation4(int device) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vecAdd;

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    for (int round = 0; round < ROUNDS; round++) {
        CUdeviceptr d_a, d_b, d_c;
        size_t size = SIZE_LARGE;
        float* h_c = (float*)malloc(size * sizeof(float));

        float a[size], b[size], c[size];

        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = i;
        }

        cuMemAlloc(&d_a, size * sizeof(float));
        cuMemAlloc(&d_b, size * sizeof(float));
        cuMemAlloc(&d_c, size * sizeof(float));

        int one = 1;
        cuMemsetD32(d_a, one, size);
        cuMemsetD32(d_b, one, size);

        checkError(cuModuleLoad(
            &cuModule,
            "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
        checkError(
            cuModuleGetFunction(&vecAdd, cuModule, "_Z10mul_kernelPfS_S_i"));

        int n = size;
        void* args[] = {&d_a, &d_b, &d_c, &n};
        checkError(cuLaunchKernel(vecAdd, size / THREADS_PER_BLOCK, 1, 1,
                                  THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0));
        cuCtxSynchronize();

        cuMemcpyDtoH(h_c, d_c, size * sizeof(float));

        bool valid = true;
        for (int i = 0; i < size; i++) {
            if (h_c[i] != i * i) {
                valid = false;
                break;
            }
        }
        printf("Process %d: data is %svalid\n", getpid(), valid ? "" : "in");

        free(h_c);
        cuMemFree(d_a);
        cuMemFree(d_b);
        cuMemFree(d_c);
    }
    cuCtxDestroy(cuContext);

}

TEST(Stress_Large, large_data_mdev) {
    for (int i = 0; i < DEVICE_COUNT; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            run_calculation4(i);
        } else if (pid < 0) {
            printf("Error: fork failed.\n");
        }
    }

    for (int i = 0; i < DEVICE_COUNT; i++) {
        wait(NULL);
    }
}
