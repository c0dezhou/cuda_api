#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <gtest/gtest.h>

#define SIZE 100000
#define THREADS_PER_BLOCK 512

void process(float* a, float* b, float* c, int start, int end) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vecAdd;
    CUdeviceptr d_a, d_b, d_c;

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_a, (end - start) * sizeof(float));
    cuMemAlloc(&d_b, (end - start) * sizeof(float));
    cuMemAlloc(&d_c, (end - start) * sizeof(float));

    cuMemcpyHtoD(d_a, a + start, (end - start) * sizeof(float));
    cuMemcpyHtoD(d_b, b + start, (end - start) * sizeof(float));

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    cuModuleGetFunction(&vecAdd, cuModule, "_Z10add_kernelPfS_S_i");

    int size = end - start;
    void *args[] = { &d_a, &d_b, &d_c, &size };
    cuLaunchKernel(vecAdd, size / THREADS_PER_BLOCK, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();

    cuMemcpyDtoH(c + start, d_c, size * sizeof(float));

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(cuContext);
}

TEST(MPROC, single_dev_fork) {
    float a[SIZE], b[SIZE], c[SIZE];

    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = i;
    }

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "Fork failed\n");
        // return 1;
    } else if (pid == 0) {
        process(a, b, c, 0, SIZE/2);
    } else {
        process(a, b, c, SIZE/2, SIZE);
        wait(NULL); 
    }

}
