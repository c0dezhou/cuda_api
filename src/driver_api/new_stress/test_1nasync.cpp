#include "test_utils.h"

#define EPSILON 0.001f

template <typename T>
void one_s_4_e_async(int d) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));
    checkError(cuCtxSetCurrent(cuContext));

    CUmodule cuModule;
    checkError(cuModuleLoad(
        &cuModule,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction add_kernel, sub_kernel, mul_kernel, div_kernel;
    checkError(cuModuleGetFunction(&add_kernel, cuModule,
                                   "_Z18add_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&sub_kernel, cuModule,
                                   "_Z18sub_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&mul_kernel, cuModule,
                                   "_Z18mul_inplace_kernelPfS_i"));
    checkError(cuModuleGetFunction(&div_kernel, cuModule,
                                   "_Z18div_inplace_kernelPfS_i"));

    CUstream cuStream;
    checkError(cuStreamCreate(&cuStream, 0));

    int num_event = 4;
    std::vector<CUevent> startEvent(num_event), stopEvent(num_event);
    for (int i = 0; i < num_event; i++) {
        checkError(cuEventCreate(&startEvent[i], CU_EVENT_DEFAULT));
        checkError(cuEventCreate(&stopEvent[i], CU_EVENT_DEFAULT));
    }

    std::vector<CUfunction> kernels = {add_kernel, mul_kernel, sub_kernel,
                                       div_kernel};

    srand(time(NULL));

    for (int N = 1; N <= 1024 * 1024 * 1024; N++) {
        size_t size = N * sizeof(T);

        CUdeviceptr d_a;
        checkError(cuMemAlloc(&d_a, size));
        CUdeviceptr d_b;
        checkError(cuMemAlloc(&d_b, size));

        T* h_a = new T[N];
        T* h_b = new T[N];
        T* h_c = new T[N];

        for (int i = 0; i < N; i++) {
            h_a[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            h_b[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            h_c[i] = h_a[i];
        }

        checkError(cuMemcpyHtoDAsync(d_a, h_a, size, cuStream));
        checkError(cuMemcpyHtoDAsync(d_b, h_b, size, cuStream));

        void* args[] = {&d_a, &d_b, &N};
        for (int i = 0; i < num_event; i++) {
            checkError(cuEventRecord(startEvent[i], cuStream));
            checkError(cuLaunchKernel(kernels[i % kernels.size()], N / 256 + 1,
                                      1, 1, 256, 1, 1, 0, cuStream, args,
                                      nullptr));
            checkError(cuEventRecord(stopEvent[i], cuStream));

            float milliseconds = 0;
            checkError(cuEventSynchronize(stopEvent[i]));
            checkError(
                cuEventElapsedTime(&milliseconds, startEvent[i], stopEvent[i]));
            std::cout << d << " device, Time for operation " << i << " of size "
                      << N << ": " << milliseconds << " ms" << std::endl;
        }

        checkError(cuMemcpyDtoHAsync(h_a, d_a, size, cuStream));

        checkError(cuStreamSynchronize(cuStream));

        for (int i = 0; i < N; ++i) {
            if (fabs(h_a[i] - ((((h_c[i] + h_b[i]) * h_b[i]) - h_b[i]) /
                               h_b[i])) > EPSILON) {
                std::cerr << "Result verification failed at element " << i
                          << "! it's" << h_a[i] <<"\n";
                exit(EXIT_FAILURE);
            }
        }

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        checkError(cuMemFree(d_a));
        checkError(cuMemFree(d_b));
    }

    for (int i = 0; i < num_event; i++) {
        checkError(cuEventDestroy(startEvent[i]));
        checkError(cuEventDestroy(stopEvent[i]));
    }
    checkError(cuStreamDestroy(cuStream));
    checkError(cuModuleUnload(cuModule));
    checkError(cuCtxDestroy(cuContext));

    std::cout << "Test PASSED\n";
}

TEST(STRESS_NEW, 1nasync) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        one_s_4_e_async<float>(d);
    }
}
