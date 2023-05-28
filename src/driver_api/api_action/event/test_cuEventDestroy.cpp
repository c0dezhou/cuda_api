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
    // TODO: 解决
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    CUevent event_;
    cuEventCreate(&event_, CU_STREAM_DEFAULT);
    cuStreamWaitEvent(stream, event_, 0);

    // delay
    float sec = 3;
    void* args[] = {&sec};
    result =
        cuLaunchKernel(func_delay, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr);

    result = cuEventQuery(event_);
    EXPECT_EQ(CUDA_ERROR_NOT_READY, result);

    cuEventDestroy(event_);
    cuStreamDestroy(stream);
}

TEST_F(CuEventTest, AC_INV_EventDestroy_DestroyEventByAnotherStream) {
    // TODO: 解决
    // 事件不绑定到特定流
    // 它们本质上是时间标记，可以记录在一个流中，然后在任何其他流（或同一流）中等待。
    // 因此，尝试使用与记录事件不同的流来销毁事件不会报错。
    // 事件的销毁独立于流，因此可以用其他流销毁
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
