#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamWaitEvent_WaitForEventCompletion) {
    // TODO: 解决 renew res2 = 0
    // 此代码正在创建两个 CUDA 流并在每个流中启动一个内核。 CUDA
    // 事件用于同步流，这样第二个流中的操作会一直等待，直到事件被记录在第一个流中。

    // 在将结果从设备复制到主机之前，正在同步两个流。
    // 但是，第二个内核启动到第二个流 (cuStream2)，并且不能保证它在 cuMemcpyDtoH
    // 调用 result2 之前完成执行。 即使正在同步
    // cuStream2，内核执行也可能尚未完成，因为 cuStreamWaitEvent 仅确保
    // cuStream2 等待 cuStream1 中的操作完成。 它不会阻塞 CPU 线程。

    // 所以解决方案是在第二次内核启动后添加另一个事件，然后等待此事件完成，然后再将数据复制回主机：
    CUstream cuStream1, cuStream2;
    cuStreamCreate(&cuStream1, 0);
    cuStreamCreate(&cuStream2, CU_STREAM_NON_BLOCKING);

    CUevent event;
    cuEventCreate(&event, 0);

    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));
    cuMemsetD32((CUdeviceptr)d_result, 0, 1);

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream1, args, nullptr);
    cuStreamSynchronize(cuStream1);

    cuEventRecord(event, cuStream1);

    cuStreamWaitEvent(cuStream2, event, 0);

    int result2 = 0;
    int* d_result2;
    cuMemAlloc((CUdeviceptr*)&d_result2, sizeof(int));
    cuMemsetD32((CUdeviceptr)d_result2, 0, 1);

    void* args1[] = {&input_a, &input_b, &d_result2};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream2, args, nullptr);

    cuStreamSynchronize(cuStream1);
    cuStreamSynchronize(cuStream2);

    CUevent event2;
    cuEventCreate(&event2, 0);
    cuEventRecord(event2, cuStream2);
    cuEventSynchronize(event2);

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
