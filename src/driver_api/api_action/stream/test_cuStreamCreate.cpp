#include <cuda.h>
#include "gtest/gtest.h"
#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamCreate_TestDefaultFlags) {
    CUstream new_stream;
    CUresult result = cuStreamCreate(&new_stream, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_SUCCESS)
        << "cuStreamCreate failed with CU_STREAM_DEFAULT flag.";
}

TEST_F(CuStreamTests, StreamCreate_NonBlockingFlag) {
    // TODO 解决
    const int N = 1 << 20; // large enough to ensure operations take noticeable time

    // create two streams, one with the non-blocking flag
    CUstream streamDefault, streamNonBlocking;
    cuStreamCreate(&streamDefault, 0);
    cuStreamCreate(&streamNonBlocking, CU_STREAM_NON_BLOCKING);

    // allocate some memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_c, N * sizeof(float));

    // initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // copy h_a and h_b to device memory
    cuMemcpyHtoDAsync(d_a, h_a, N * sizeof(float), streamDefault);
    cuMemcpyHtoDAsync(d_b, h_b, N * sizeof(float), streamNonBlocking);

    // launch kernels on different streams
    void* args1[] = { &d_a, &d_b, &d_c, &N };
    cuLaunchKernel(cuFunction, N / 256, 1, 1, 256, 1, 1, 0, streamDefault, args1, nullptr);

    void* args2[] = { &d_b, &d_a, &d_c, &N };
    cuLaunchKernel(cuFunction, N / 256, 1, 1, 256, 1, 1, 0, streamNonBlocking, args2, nullptr);

    // copy back to host memory
    cuMemcpyDtoHAsync(h_c, d_c, N * sizeof(float), streamDefault);

    // synchronization
    cuStreamSynchronize(streamDefault);
    cuStreamSynchronize(streamNonBlocking);

    // check results
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_c[i], h_a[i] + h_b[i]);
    }

    // cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);

    cuStreamDestroy(streamDefault);
    cuStreamDestroy(streamNonBlocking);
}


TEST_F(CuStreamTests, AC_INV_StreamCreate_InvalidPhStreamPointer) {
    CUstream* invalid_stream_ptr = nullptr;
    CUresult result = cuStreamCreate(invalid_stream_ptr, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE)
        << "cuStreamCreate should have failed with "
           "an invalid phStream pointer.";
}
