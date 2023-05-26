#include "event_tests.h"

TEST_F(CuEventTest, AC_BA_EventSynchronize_SyncCompletedEvent) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(event, stream);
    float sec = 3;
    void* args[] = {&sec};
    result =
        cuLaunchKernel(func_delay, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr);

    cuStreamSynchronize(stream);

    result = cuEventSynchronize(event);
    EXPECT_EQ(CUDA_SUCCESS, result);
    cuStreamDestroy(stream);
}

TEST_F(CuEventTest, AC_INV_EventSynchronize_SyncIncompleteEvent) {
    // TODO: 解决，正在尝试同步尚未创建或记录的事件，这会导致未定义的行为。
    GTEST_SKIP();  // due to core dump
    CUevent event_;
    CUstream stream;
    cuStreamCreate(&stream, 0);
    CUresult res = cuEventSynchronize(event_);
    EXPECT_EQ(res, 400);
    res = cuStreamDestroy(stream);
    EXPECT_EQ(res, 400);
}

TEST_F(CuEventTest, AC_INV_EventSynchronize_SyncDestroyedEvent) {
    // TODO: 解决 未定义行为
    cuEventDestroy(event);

    result = cuEventSynchronize(event);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_INV_EventSynchronize_SyncInvalidEvent) {
    GTEST_SKIP();
    CUevent invalid_event = reinterpret_cast<CUevent>(0xdeadbeef);

    result = cuEventSynchronize(invalid_event);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_BA_EventSynchronize_SyncDifferentContext) {
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
    result = cuEventSynchronize(event);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);

    cuCtxDestroy(cuContext1);
    cuCtxDestroy(cuContext2);
}

TEST_F(CuEventTest, AC_INV_EventSynchronize_SyncBeforeRecord) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventSynchronize(event);
    cuStreamDestroy(stream);
}