#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamWaitEvent_WaitForEventCompletion) {
    CUstream cuStream1, cuStream2;
    cuStreamCreate(&cuStream1, 0);
    cuStreamCreate(&cuStream2, CU_STREAM_NON_BLOCKING);

    CUevent event;
    cuEventCreate(&event, 0);

    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream1, args, nullptr);

    cuEventRecord(event, cuStream1);

    cuStreamWaitEvent(cuStream2, event, 0);

    int result2 = 0;
    int* d_result2;
    cuMemAlloc((CUdeviceptr*)&d_result2, sizeof(int));

    void* args1[] = {&input_a, &input_b, &d_result2};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream2, args, nullptr);

    cuStreamSynchronize(cuStream1);
    cuStreamSynchronize(cuStream2);

    cuMemcpyDtoH(&result, (CUdeviceptr)d_result, sizeof(int));
    cuMemcpyDtoH(&result2, (CUdeviceptr)d_result2, sizeof(int));

    EXPECT_EQ(result, input_a + input_b);
    EXPECT_EQ(result2, input_a + input_b);

    cuMemFree((CUdeviceptr)d_result);
    cuMemFree((CUdeviceptr)d_result2);
    cuEventDestroy(event);
    cuStreamDestroy(cuStream1);
    cuStreamDestroy(cuStream2);
}

TEST_F(CuStreamTests, AC_INV_StreamWaitEvent_EventNotRecorded) {
    CUstream cuStream;
    cuStreamCreate(&cuStream, 0);

    CUevent event;
    cuEventCreate(&event, 0);

    CUresult result = cuStreamWaitEvent(cuStream, event, 0);

    EXPECT_EQ(result, CUDA_SUCCESS);

    cuEventDestroy(event);
    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_INV_StreamWaitEvent_FlagsMustBeZero) {
    CUstream cuStream;
    cuStreamCreate(&cuStream, 0);
    CUevent event;
    cuEventCreate(&event, 0);
    cuEventRecord(event, cuStream);
    CUresult result = cuStreamWaitEvent(cuStream, event, 1);
    EXPECT_NE(result, CUDA_SUCCESS);
    cuEventDestroy(event);
    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_BA_StreamWaitEvent_CheckEventCompletion) {
    CUstream cuStream;
    cuStreamCreate(&cuStream, 0);
    CUevent event;
    cuEventCreate(&event, 0);
    cuEventRecord(event, cuStream);
    CUresult query_result = cuEventQuery(event);
    EXPECT_EQ(query_result, CUDA_ERROR_NOT_READY);
    cuStreamSynchronize(cuStream);
    query_result = cuEventQuery(event);
    EXPECT_EQ(query_result, CUDA_SUCCESS);
    cuEventDestroy(event);
    cuStreamDestroy(cuStream);
}
