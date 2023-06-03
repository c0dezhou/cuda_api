#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>

#define SIZE 100000000  // 100 million elements, roughly ~400MB if integers
#define THREADS_PER_BLOCK 512

int main() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUdeviceptr d_a, d_b, d_c;
    CUmodule cuModule;
    CUfunction vecAdd;

    float *h_a = (float*) malloc(SIZE * sizeof(float));
    float *h_b = (float*) malloc(SIZE * sizeof(float));
    float *h_c = (float*) malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_a, SIZE * sizeof(float));
    cuMemAlloc(&d_b, SIZE * sizeof(float));
    cuMemAlloc(&d_c, SIZE * sizeof(float));

    cuMemcpyHtoD(d_b, h_b, SIZE * sizeof(float));
    cuMemcpyHtoD(d_a, h_a, SIZE * sizeof(float));

    cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx");
    cuModuleGetFunction(&vecAdd, cuModule, "_Z10add_kernelPfS_S_i");

    int size = SIZE;
    void* args[] = {&d_a, &d_b, &d_c, &size};
    cuLaunchKernel(vecAdd, size / THREADS_PER_BLOCK, 1, 1, THREADS_PER_BLOCK, 1,
                   1, 0, 0, args, 0);
    cuCtxSynchronize();

    cuMemcpyDtoH(h_c, d_c, SIZE * sizeof(float));

    bool valid = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_c[i] != 2.0) {
            valid = false;
            break;
        }
    }
    printf("Process %d: data is %svalid\n", getpid(), valid ? "" : "in");

    free(h_a);
    free(h_b);
    free(h_c);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(cuContext);

    return 0;
}
