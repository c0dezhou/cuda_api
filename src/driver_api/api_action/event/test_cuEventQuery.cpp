#include "event_tests.h"

TEST_F(CuEventTest, AC_BA_EventQuery_QueryCompletedEvent) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(event, stream);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    result = cuEventQuery(event);
    EXPECT_EQ(CUDA_SUCCESS, result);
}

TEST_F(CuEventTest, AC_INV_EventQuery_QueryIncompleteEvent) {
    // TODO: 待确认
    GTEST_SKIP(); // due to core dump
    CUevent event_;
    result = cuEventQuery(event_);
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, result);
}

TEST_F(CuEventTest, AC_INV_EventQuery_QueryDestroyedEvent) {
    // TODO: 待确认
    cuEventDestroy(event);
    result = cuEventQuery(event);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_INV_EventQuery_QueryInvalidEvent) {
    GTEST_SKIP();
    CUevent invalid_event = reinterpret_cast<CUevent>(0xdeadbeef);

    result = cuEventQuery(invalid_event);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_BA_EventQuery_QueryDifferentContext) {
    CUdevice cuDevice1, cuDevice2;
    CUcontext cuContext1, cuContext2;
    cuDeviceGet(&cuDevice1, 0);
    cuCtxCreate(&cuContext1, 0, cuDevice1);

    cuDeviceGet(&cuDevice2, 1);
    cuCtxCreate(&cuContext2, 0, cuDevice2);

    cuCtxSetCurrent(cuContext1);
    CUstream stream1;
    cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    cuEventRecord(event, stream1);
    float sec = 3;
    void* args[] = {&sec};
    result =
        cuLaunchKernel(func_delay, 1, 1, 1, 1, 1, 1, 0, stream1, args, nullptr);

    cuCtxSetCurrent(cuContext2);
    result = cuEventQuery(event);
    EXPECT_EQ(CUDA_SUCCESS, result);

    cuCtxDestroy(cuContext1);
    cuCtxDestroy(cuContext2);
}

TEST_F(CuEventTest, AC_INV_EventQuery_QueryBeforeRecord) {
    result = cuEventQuery(event);
    EXPECT_EQ(CUDA_SUCCESS, result);
}

TEST_F(CuEventTest, AC_BA_EventQuery_QueryInloop) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(event, stream);

    result;
    do
    {
            // std::cout << "yes" << std::endl;
            // 由于异步，不阻塞host端，可以在等待事件完成时做别的任务
            result = cuEventQuery(event);
    } while (result == CUDA_ERROR_NOT_READY);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    EXPECT_EQ(CUDA_SUCCESS, result);
}

TEST_F(CuEventTest, AC_INV_EventQuery_QueryDifferentStatus) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(event, stream);
    for (int i = 0; i < 3; i++)
    {
        result = cuEventQuery(event);

        if (i == 0)
        {
            EXPECT_EQ(CUDA_ERROR_NOT_READY, result);
        }
        else
        {
            EXPECT_EQ(CUDA_SUCCESS, result);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    EXPECT_EQ(CUDA_SUCCESS, result);
}
