// 使用多个子进程对内存中的共享数组执行向量乘法。 每个子进程在数组的不同部分执行计算。
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <cuda.h>

#define CUDA_CALL(call) \
do { \
    CUresult cuResult = call; \
    if (cuResult != CUDA_SUCCESS) { \
        const char *errName = nullptr, *errStr = nullptr; \
        cuGetErrorName(cuResult, &errName); \
        cuGetErrorString(cuResult, &errStr); \
        std::cerr << "CUDA error: " << errName << " (" << errStr << ") at " << #call << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define N 1024
#define BLOCK_SIZE 256
#define NUM_PROCESSES 4
#define NUM_ELEMS_PER_PROCESS (N / NUM_PROCESSES)

void process(float *data, int offset, int num_elems) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGet(&device, 0));
    CUDA_CALL(cuCtxCreate(&context, 0, device));
    CUDA_CALL(cuModuleLoad(&module, "C:\\Users\\zhouf\\Desktop\\cuda_workspace\\cuda_api\\common\\cuda_kernel\\cuda_kernel.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernelFunc, module, "_Z14vec_multiply_2Pfi"));

    int blockSize = BLOCK_SIZE;
    int gridSize = (num_elems + blockSize - 1) / blockSize;

    float *d_data;
    CUDA_CALL(cuMemAlloc((void**)&d_data, num_elems * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoD(d_data, data + offset, num_elems * sizeof(float)));

    void *args[] = { &d_data, &num_elems };
    CUDA_CALL(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL));
    CUDA_CALL(cuCtxSynchronize());

    CUDA_CALL(cuMemcpyDtoH(data + offset, d_data, num_elems * sizeof(float)));

    CUDA_CALL(cuMemFree(d_data));
    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    float *data = (float*)mmap(NULL, N * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    for (int i = 0; i < NUM_PROCESSES; ++i) {
        pid_t pid = fork();

        if (pid == 0) { // child process
            process(data, i * NUM_ELEMS_PER_PROCESS, NUM_ELEMS_PER_PROCESS);
            exit(0);
        } else if (pid < 0) {
            std::cerr << "Fork failed\n";
            return 1;
        }
    }

    for (int i = 0; i < NUM_PROCESSES; ++i) {
        wait(NULL); // wait for child processes to finish
    }

    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << "\n";
    }

    munmap(data, N * sizeof(float));

    return 0;
}
