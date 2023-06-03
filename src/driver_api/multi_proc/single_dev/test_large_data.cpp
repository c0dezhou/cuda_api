#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#define SIZE 100000000  // ~400MB (int)
#define THREADS_PER_BLOCK 512


void process() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vecAdd;
    CUdeviceptr d_a, d_b, d_c;
    float *h_c = (float*) malloc(SIZE * sizeof(float));

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_a, SIZE * sizeof(float));
    cuMemAlloc(&d_b, SIZE * sizeof(float));
    cuMemAlloc(&d_c, SIZE * sizeof(float));

    int one = 1;
    cuMemsetD32(d_a, one, SIZE);
    cuMemsetD32(d_b, one, SIZE);

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    cuModuleGetFunction(&vecAdd, cuModule, "_Z10add_kernelPfS_S_i");

    int n = SIZE;
    void* args[] = {&d_a, &d_b, &d_c, &n};
    cuLaunchKernel(vecAdd, SIZE / THREADS_PER_BLOCK, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();
    
    cuMemcpyDtoH(h_c, d_c, SIZE * sizeof(float));

    bool valid = true;
    for (int i = 0; i < SIZE; i++) {
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

TEST(MPROC, single_dev_large_data) {
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "Fork failed\n");
        // return 1;
    } else if (pid == 0) {
        process();
    } else {
        process();
        wait(NULL);
    }

}
