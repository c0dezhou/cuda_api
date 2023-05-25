#include "test_utils.h"

const int NUM_TASKS = 4;
int NUM_ELEMENTS = 1000000;
const int BLOCK_SIZE = 256;

CUstream single_stream;

#define init_cuda() \
    checkError(cuInit(0));\
    CUdevice device;\
    checkError(cuDeviceGet(&device, 0));\
    CUcontext context; \
    checkError(cuCtxCreate(&context, 0, device)); \
    checkError(cuStreamCreate(&single_stream, CU_STREAM_NON_BLOCKING));

#define cleanup_cuda()                          \
    checkError(cuStreamDestroy(single_stream)); \
    checkError(cuCtxDestroy(context));


void init_host_memory(float** a, float** b, float** c, int n) {
    checkError(
        cuMemHostAlloc((void**)a, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
    checkError(
        cuMemHostAlloc((void**)b, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
    checkError(
        cuMemHostAlloc((void**)c, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));

    for (int i = 0; i < n; i++) {
        (*a)[i] = i;
        (*b)[i] = i + 1;
        (*c)[i] = 0;
    }
}

void free_host_memory(float* a, float* b, float* c) {
    checkError(cuMemFreeHost(a));
    checkError(cuMemFreeHost(b));
    checkError(cuMemFreeHost(c));
}

void init_device_memory(float** d_a, float** d_b, float** d_c, int n) {
    checkError(cuMemAlloc((CUdeviceptr*)d_a, n * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)d_b, n * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)d_c, n * sizeof(float)));
}

void free_device_memory(float* d_a, float* d_b, float* d_c) {
    checkError(cuMemFree((CUdeviceptr)d_a));
    checkError(cuMemFree((CUdeviceptr)d_b));
    checkError(cuMemFree((CUdeviceptr)d_c));
}

void check_result(float* a,
                  float* b,
                  float* c,
                  int n,
                  const char* kernel_name) {
    const float epsilon = 1e-6f;

    for (int i = 0; i < n; i++) {
        float expected;
        if (strcmp(kernel_name, "add_kernel") == 0) {
            expected = a[i] + b[i];
        } else if (strcmp(kernel_name, "mul_kernel") == 0) {
            expected = a[i] * b[i];
        } else if (strcmp(kernel_name, "sub_kernel") == 0) {
            expected = a[i] - b[i];
        } else if (strcmp(kernel_name, "div_kernel") == 0) {
            expected = a[i] / b[i];
        } else {
            std::cerr << "Unknown kernel name: " << kernel_name << std::endl;
            exit(1);
        }

        if (fabs(c[i] - expected) > epsilon) {
            std::cerr << "Error: result mismatch at index " << i << std::endl;
            std::cerr << "Actual: " << c[i] << ", Expected: " << expected
                      << std::endl;
            exit(1);
        }
    }

    std::cout << "Success: result matched for " << kernel_name << std::endl;
}


TEST(COMBINE, 1_stream_n_event) {
    init_cuda();

    float *h_a[NUM_TASKS], *h_b[NUM_TASKS],
        *h_c[NUM_TASKS];
    float *d_a[NUM_TASKS], *d_b[NUM_TASKS],
        *d_c[NUM_TASKS];
    for (int i = 0; i < NUM_TASKS; i++) {
        init_host_memory(&h_a[i], &h_b[i], &h_c[i], NUM_ELEMENTS);
        init_device_memory(&d_a[i], &d_b[i], &d_c[i], NUM_ELEMENTS);
    }

    for (int i = 0; i < NUM_TASKS; i++) {
        checkError(cuMemcpyHtoDAsync((CUdeviceptr)(d_a[i]), h_a[i],
                                     NUM_ELEMENTS * sizeof(float),
                                     single_stream));
        checkError(cuMemcpyHtoDAsync((CUdeviceptr)(d_b[i]), h_b[i],
                                     NUM_ELEMENTS * sizeof(float),
                                     single_stream));

        CUfunction kernel, add_kernel, mul_kernel, sub_kernel, div_kernel;
        const char* kernel_name;
        CUmodule module;
        CUresult result =
            cuModuleLoad(&module,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "cuda_kernel.ptx");

        result =
            cuModuleGetFunction(&add_kernel, module, "_Z10add_kernelPfS_S_i");
        result =
            cuModuleGetFunction(&mul_kernel, module, "_Z10mul_kernelPfS_S_i");
        result =
            cuModuleGetFunction(&sub_kernel, module, "_Z10sub_kernelPfS_S_i");
        result =
            cuModuleGetFunction(&div_kernel, module, "_Z10div_kernelPfS_S_i");

        switch (i) {
            case 0:
                kernel = add_kernel;
                kernel_name = "add_kernel";
                break;
            case 1:
                kernel = mul_kernel;
                kernel_name = "mul_kernel";
                break;
            case 2:
                kernel = sub_kernel;
                kernel_name = "sub_kernel";
                break;
            case 3:
                kernel = div_kernel;
                kernel_name = "div_kernel";
                break;
            default:
                std::cerr << "Invalid task id: " << i << std::endl;
                exit(1);
        }

        int grid_size = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        void* kernel_args[] = {&d_a[i], &d_b[i], &d_c[i], &NUM_ELEMENTS};
        checkError(cuLaunchKernel(kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0,
                                  single_stream, kernel_args, NULL));

        checkError(cuMemcpyDtoHAsync(h_c[i], (CUdeviceptr)(d_c[i]),
                                     NUM_ELEMENTS * sizeof(float),
                                     single_stream));

        CUevent event;
        checkError(cuEventCreate(&event, CU_EVENT_DEFAULT));
        checkError(cuEventRecord(event, single_stream));

        checkError(cuEventSynchronize(event));

        check_result(h_a[i], h_b[i], h_c[i], NUM_ELEMENTS, kernel_name);

        checkError(cuEventDestroy(event));
    }
    cleanup_cuda();
}
