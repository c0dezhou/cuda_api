#include "test_utils.h"

// Sizes for small, medium, and large datasets
#define size_SMALL 10000
#define size_MEDIUM 10000000
#define size_LARGE 100000000
#define THREADS_PER_BLOCK 512
#define DEVICE_COUNT 2

void run_calculation1(int device, size_t size) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vecAdd;
    CUdeviceptr d_a, d_b, d_c;
    float* h_c = (float*)malloc(size * sizeof(float));

    float a[size], b[size], c[size];

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_a, size * sizeof(float));
    cuMemAlloc(&d_b, size * sizeof(float));
    cuMemAlloc(&d_c, size * sizeof(float));

    int one = 1;
    cuMemsetD32(d_a, one, size);
    cuMemsetD32(d_b, one, size);

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    cuModuleGetFunction(&vecAdd, cuModule, "_Z10add_kernelPfS_S_i");

    int n = size;
    void* args[] = {&d_a, &d_b, &d_c, &n};
    cuLaunchKernel(vecAdd, size / THREADS_PER_BLOCK, 1, 1, THREADS_PER_BLOCK, 1,
                   1, 0, 0, args, 0);
    cuCtxSynchronize();

    cuMemcpyDtoH(h_c, d_c, size * sizeof(float));

    bool valid = true;
    for (int i = 0; i < size; i++) {
        if (h_c[i] != 2) {
            valid = false;
            break;
        }
    }
    printf("Process %d: data is %svalid\n", getpid(), valid ? "" : "in");

    free(h_c);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(cuContext);
}

TEST(MPROC, multi_dev_fork) {
    int deviceCount;
    cuInit(0);
    cuDeviceGetCount(&deviceCount);
    if (deviceCount < DEVICE_COUNT) {
        printf("Error: requires at least %d GPUs.\n", DEVICE_COUNT);
    }

    size_t sizes[] = {size_SMALL, size_MEDIUM, size_LARGE};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < DEVICE_COUNT; i++) {
        for (int j = 0; j < num_sizes; j++) {
            pid_t pid = fork();
            if (pid == 0) {
                run_calculation1(i, sizes[j]);
            } else if (pid < 0) {
                printf("Error: fork failed.\n");
            }
        }
    }

    for (int i = 0; i < DEVICE_COUNT * num_sizes; i++) {
        wait(NULL);
    }

}
