#include "stream_tests.h"

TEST_F(CuStreamTests, AC_INV_StreamQuery_EmptyStream) {
    CUstream cuStream;
    cuStreamCreate(&cuStream, 0);

    CUresult result = cuStreamQuery(cuStream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_BA_StreamQuery_PendingOperations) {
    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream, args, nullptr);

    cuMemcpyDtoHAsync(&result, (CUdeviceptr)d_result, sizeof(int), cuStream);

    CUresult res = cuStreamQuery(cuStream);
    EXPECT_EQ(res, CUDA_SUCCESS);

    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_BA_StreamQuery_CompletedOperations) {
    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream, args, nullptr);

    cuMemcpyDtoHAsync(&result, (CUdeviceptr)d_result, sizeof(int), cuStream);

    cuStreamSynchronize(cuStream);

    CUresult queryResult = cuStreamQuery(cuStream);
    EXPECT_EQ(queryResult, CUDA_SUCCESS);

    cuMemFree((CUdeviceptr)d_result);
}