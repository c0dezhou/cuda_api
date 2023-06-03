#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define SIZE_SMALL 10000
#define SIZE_MEDIUM 10000000
#define SIZE_LARGE 100000000
#define THREADS_PER_BLOCK 512
#define ROUNDS 10

void run_calculation(int device, size_t size_n) {
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
        float* h_data = (float*)malloc(size_n * sizeof(float));

        for (int i = 0; i < size_n; i++) {
            h_data[i] = i;
        }

        cuMemAlloc(&d_data, size_n * sizeof(float));

        cuMemcpyHtoD(d_data, h_data, size_n * sizeof(float));

        void* args[] = {&d_data, &size_n};
        cuLaunchKernel(cuFunction, size_n / THREADS_PER_BLOCK, 1, 1,
                       THREADS_PER_BLOCK, 1, 1, 0, NULL, args, NULL);
        cuCtxSynchronize();

        cuMemcpyDtoH(h_data, d_data, size_n * sizeof(float));

        cuMemFree(d_data);

        for (int i = 0; i < size_n; i++) {
            if (h_data[i] != i * 2.0f) {
                printf("Error: data[%.2f] = %.2f\n", i, h_data[i]);
                exit(-1);
            }
        }

        free(h_data);
    }

    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);
}

int main(int argc, char** argv) {
    printf("Running sub process...\n");
    if (argc != 3) {
        printf("Error: requires 2 arguments.\n");
        return -1;
    }

    int device = atoi(argv[1]);
    size_t size;
    if (strcmp(argv[2], "small") == 0) {
        size = SIZE_SMALL;
    } else if (strcmp(argv[2], "medium") == 0) {
        size = SIZE_MEDIUM;
    } else if (strcmp(argv[2], "large") == 0) {
        size = SIZE_LARGE;
    } else {
        printf("Error: invalid size argument.\n");
        return -1;
    }

    run_calculation(device, size);
    printf("Finish running sub process...\n");

    return 0;
}
