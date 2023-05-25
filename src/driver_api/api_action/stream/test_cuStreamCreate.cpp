#include <cuda.h>
#include "gtest/gtest.h"
#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamCreate_TestDefaultFlags) {
    CUstream new_stream;
    CUresult result = cuStreamCreate(&new_stream, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_SUCCESS)
        << "cuStreamCreate failed with CU_STREAM_DEFAULT flag.";
}

TEST_F(CuStreamTests, AC_BA_StreamCreate_TestNonBlockingFlag) {
    // TODO：待确认
    // TODO: redesgin this case
    
    // CUstream new_stream;
    // CUresult result = cuStreamCreate(&new_stream, CU_STREAM_NON_BLOCKING);

    // EXPECT_EQ(result, CUDA_SUCCESS)
    //     << "cuStreamCreate failed with CU_STREAM_NON_BLOCKING flag.";

    int N = 1024;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 1.0f);
    std::vector<float> h_C(N, 0.0f);

    float *d_A, *d_B, *d_C;
    cuMemAlloc((CUdeviceptr*)&d_A, bytes);
    cuMemAlloc((CUdeviceptr*)&d_B, bytes);
    cuMemAlloc((CUdeviceptr*)&d_C, bytes);

    cuMemcpyHtoD((CUdeviceptr)d_A, h_A.data(), bytes);
    cuMemcpyHtoD((CUdeviceptr)d_B, h_B.data(), bytes);

    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    void* kernel_args[] = {&d_A, &d_B, &d_C, &N};

    CUfunction vector_add_kernel;

    CUresult result = cuModuleGetFunction(&vector_add_kernel, cuModule,
                                         "_Z6vecAddPfS_S_");
    EXPECT_EQ(result, CUDA_SUCCESS);

    // cuStreamSynchronize(new_stream);

    cuMemcpyDtoH(h_C.data(), (CUdeviceptr)d_C, bytes);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(h_C[i], 2.0f) << "Result mismatch at index " << i;
    }

    // cuStreamDestroy(new_stream);
    cuMemFree((CUdeviceptr)d_A);
    cuMemFree((CUdeviceptr)d_B);
    cuMemFree((CUdeviceptr)d_C);
}

TEST_F(CuStreamTests, AC_INV_StreamCreate_InvalidPhStreamPointer) {
    CUstream* invalid_stream_ptr = nullptr;
    CUresult result = cuStreamCreate(invalid_stream_ptr, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE)
        << "cuStreamCreate should have failed with "
           "an invalid phStream pointer.";
}
