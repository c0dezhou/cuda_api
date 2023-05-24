#include <iostream>
#include <cuda.h>
#include <unistd.h>
#include <sys/wait.h>

#define CUDA_CALL(call) \
do { \
    CUresult error = call; \
    if (error != CUDA_SUCCESS) { \
        std::cerr << "CUDA error: " << error << " at " << #call << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void addKernel(float* c, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void gpu_work(int device_id, float* a, float* b, int size) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr a_dev, b_dev, c_dev;

    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGet(&device, device_id));
    CUDA_CALL(cuCtxCreate(&context, 0, device));

    // (Replace the path to the ptx file)
    CUDA_CALL(cuModuleLoad(&module, "path/to/kernel.ptx"));
    CUDA_CALL(cuModuleGetFunction(&kernel, module, "addKernel"));

    // Assuming 'size' is the size of your data and 'a' and 'b' are your input arrays
    CUDA_CALL(cuMemAlloc(&a_dev, size * sizeof(float)));
    CUDA_CALL(cuMemAlloc(&b_dev, size * sizeof(float)));
    CUDA_CALL(cuMemAlloc(&c_dev, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoD(a_dev, a, size * sizeof(float)));
    CUDA_CALL(cuMemcpyHtoD(b_dev, b, size * sizeof(float)));

    void* args[] = { &c_dev, &a_dev, &b_dev, &size };
    CUDA_CALL(cuLaunchKernel(kernel, size, 1, 1, 1, 1, 1, 0, NULL, args, NULL));

    // Retrieve and check results here
    // Allocate host memory for results
    float* c = new float[size];

    // Retrieve the results
    CUDA_CALL(cuMemcpyDtoH(c, c_dev, size * sizeof(float)));

    // Check the results
    bool result_correct = true;
    for (int i = 0; i < size; i++) {
        // Here, we expect c[i] to be a[i] + b[i]
        if (fabs(c[i] - (a[i] + b[i])) > 1e-7) {
            result_correct = false;
            break;
        }
    }

    if (result_correct) {
        std::cout << "Results are correct for GPU " << device_id << std::endl;
    } else {
        std::cout << "Results are incorrect for GPU " << device_id << std::endl;
    }

    delete[] c;
    CUDA_CALL(cuMemFree(a_dev));
    CUDA_CALL(cuMemFree(b_dev));
    CUDA_CALL(cuMemFree(c_dev));

    CUDA_CALL(cuCtxDestroy(context));
}

int main() {
    pid_t pid;
    int deviceCount;
    float* a = new float[size];
    float* b = new float[size];

    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGetCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            gpu_work(i, a, b, size);
            exit(0);
        } else if (pid < 0) {
            std::cerr << "Fork failed!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < deviceCount; i++) {
        wait(NULL);
    }

    delete[] a;
    delete[] b;

    return EXIT_SUCCESS;
}
