#include "test_utils.h"

#define SIZE_LARGE 1000000000
#define THREADS_PER_BLOCK 512
#define ROUNDS 10
#define DEVICE_COUNT 2

void run_calsdlation4(int device) {
    SDdevice sdDevice;
    SDcontext sdContext;
    SDmodule sdModule;
    SDfunction vecAdd;

    sdInit(0);
    sdDeviceGet(&sdDevice, 0);
    sdCtxCreate(&sdContext, 0, sdDevice);

    for (int round = 0; round < ROUNDS; round++) {
        SDdeviceptr d_a, d_b, d_c;
        size_t size = SIZE_LARGE;
        float* h_c = (float*)malloc(size * sizeof(float));

        float a[size], b[size], c[size];

        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = i;
        }

        sdMemAlloc(&d_a, size * sizeof(float));
        sdMemAlloc(&d_b, size * sizeof(float));
        sdMemAlloc(&d_c, size * sizeof(float));

        int one = 1;
        sdMemsetD32(d_a, one, size);
        sdMemsetD32(d_b, one, size);

        checkError(sdModuleLoad(
            &sdModule,
            "/data/system/yunfan/sdda_api/common/sdda_kernel/sdda_kernel.ptx"));
        checkError(
            sdModuleGetFunction(&vecAdd, sdModule, "_Z10mul_kernelPfS_S_i"));

        int n = size;
        void* args[] = {&d_a, &d_b, &d_c, &n};
        checkError(sdLaunchKernel(vecAdd, size / THREADS_PER_BLOCK, 1, 1,
                                  THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0));
        sdCtxSynchronize();

        sdMemcpyDtoH(h_c, d_c, size * sizeof(float));

        bool valid = true;
        for (int i = 0; i < size; i++) {
            if (h_c[i] != i * i) {
                valid = false;
                break;
            }
        }
        printf("Process %d: data is %svalid\n", getpid(), valid ? "" : "in");

        free(h_c);
        sdMemFree(d_a);
        sdMemFree(d_b);
        sdMemFree(d_c);
    }
    sdCtxDestroy(sdContext);

}

TEST(Stress_Large, large_data_mdev) {
    for (int i = 0; i < DEVICE_COUNT; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            run_calsdlation4(i);
        } else if (pid < 0) {
            printf("Error: fork failed.\n");
        }
    }

    for (int i = 0; i < DEVICE_COUNT; i++) {
        wait(NULL);
    }
}
