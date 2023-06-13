#include "test_utils.h"

#define EPSILON 0.001f

template <typename T>
void d2d_and_set(int d) {
    checkError(cuInit(0));

    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

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

    const int numStreams = 20;
    std::vector<CUstream> cuStreams(numStreams);
    for (int i = 0; i < numStreams; i++) {
        checkError(cuStreamCreate(&cuStreams[i], 0));
    }

    std::vector<CUevent> startEvent(numStreams), stopEvent(numStreams);
    for (int i = 0; i < numStreams; i++) {
        checkError(cuEventCreate(&startEvent[i], CU_EVENT_DEFAULT));
        checkError(cuEventCreate(&stopEvent[i], CU_EVENT_DEFAULT));
    }

    for (int N = 1; N <= 1024; N++) {
        size_t size = N * sizeof(T);

        CUdeviceptr d_a;
        checkError(cuMemAlloc(&d_a, size));
        CUdeviceptr d_b;
        checkError(cuMemAlloc(&d_b, size));
        CUdeviceptr d_c;
        checkError(cuMemAlloc(&d_c, size));

        unsigned int memset_value = 0;
        checkError(
            cuMemsetD32(d_a, memset_value, N));

        std::vector<T> h_a(N, 1.0f);
        std::vector<T> h_b(N, 2.0f);
        std::vector<T> h_c(N, 0.0f);

        checkError(cuMemcpyHtoD(d_b, h_b.data(), size)); 
        checkError(cuMemcpyHtoD(d_c, h_c.data(), size));
        checkError(cuMemcpyDtoD(d_a, d_b,
                                size));

        void* args1[] = {&d_a, &d_b, &N}; // 2 2
        void* args2[] = {&d_c, &d_b, &N}; // 0 2

        for (int i = 0; i < numStreams; i++) {
            checkError(cuEventRecord(startEvent[i], cuStreams[i]));
            checkError(cuLaunchKernel(add_kernel, N / 256 +1, 1, 1, 256, 1, 1, 0,
                                      cuStreams[i], args1, nullptr)); // 4 6 8
            checkError(cuMemcpyDtoD(
                d_c, d_a, size));  // 4 6 8
            checkError(cuLaunchKernel(sub_kernel, N / 256 +1, 1, 1, 256, 1, 1, 0,
                                      cuStreams[i], args2, nullptr)); // 2 4 6
            checkError(cuEventRecord(stopEvent[i], cuStreams[i]));
        }

        for (int i = 0; i < numStreams; i++) {
            cuStreamSynchronize(cuStreams[i]);
            float milliseconds = 0;
            checkError(cuEventSynchronize(stopEvent[i]));
            checkError(
                cuEventElapsedTime(&milliseconds, startEvent[i], stopEvent[i]));
            std::cout << "Time for operation " << i << " of size " << N << ": "
                      << milliseconds << " ms" << std::endl;
        }

        checkError(cuMemcpyDtoH(h_c.data(), d_c, size));
        
         for (int i = 0; i < N; i++) {
            if (fabs(h_c[i] - (numStreams*2)) >
                EPSILON) {  // Check the result with a small error tolerance
                std::cerr << "Verification failed at position " << i << ": "
                          << h_c[i] << " != " << numStreams * 2
                          << std::endl;
                exit(1);
            }
        }

        checkError(cuMemFree(d_a));
        checkError(cuMemFree(d_b));
        checkError(cuMemFree(d_c));
    }

    for (int i = 0; i < numStreams; i++) {
        checkError(cuEventDestroy(startEvent[i]));
        checkError(cuEventDestroy(stopEvent[i]));
        checkError(cuStreamDestroy(cuStreams[i]));
    }
    checkError(cuModuleUnload(cuModule));
    checkError(cuCtxDestroy(cuContext));
}

TEST(STRESS_NEW, d2d_set_sync) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        d2d_and_set<float>(d);
    }
}