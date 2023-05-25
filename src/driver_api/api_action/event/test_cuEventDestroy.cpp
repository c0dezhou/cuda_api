#include "event_tests.h"

TEST_F(CuEventTest, AC_BA_EventDestroy_DestroyValidEvent) {
    CUevent event_;
    cuEventCreate(&event_, 0);
    result = cuEventDestroy(event_);
    EXPECT_EQ(CUDA_SUCCESS, result);
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyInvalidEvent) {
    CUevent event_ = nullptr;
    EXPECT_EQ(cuEventDestroy(event_), CUDA_ERROR_INVALID_HANDLE);
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyConsumedEvent) {
    GTEST_SKIP(); // due to core dump
    CUevent event_;
    cuEventCreate(&event_, CU_EVENT_DEFAULT);
    std::thread destroyThread(&CuEventTest::destroyEvent, &event_);
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));

    result = cuEventSynchronize(event_);
    EXPECT_EQ(result, CUDA_ERROR_CONTEXT_IS_DESTROYED);

    destroyThread.join();
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyDestroyedEvent) {
    CUevent event_;
    cuEventCreate(&event_, 0);
    cuEventDestroy(event_);

    result = cuEventDestroy(event_);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyEventUsedInWait) {
    // TODO: 待确认
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    CUevent event_;
    cuEventCreate(&event_, 0);
    cuStreamWaitEvent(stream, event_, 0);

    result = cuEventDestroy(event_);
    EXPECT_EQ(CUDA_ERROR_NOT_READY, result);

    cuStreamDestroy(stream);
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyEventByAnotherStream) {
    // TODO: 待确认
    CUstream stream1, stream2;

    cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING);
    CUevent event_;
    cuEventCreate(&event_, 0);
    cuEventRecord(event_, stream1);
    float sec = 3;
    void* args[] = {&sec};
    result =
        cuLaunchKernel(func_delay, 1, 1, 1, 1, 1, 1, 0, stream1, args, nullptr);
    std::cout << result << std::endl;
    cuStreamWaitEvent(stream1, event_, 0);
    result = cuEventDestroy(event_);
    EXPECT_EQ(result, CUDA_ERROR_NOT_READY);
    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
}
