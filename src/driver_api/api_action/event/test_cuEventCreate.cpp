#include "event_tests.h"

TEST_F(CuEventTest, AC_BA_EventCreate_DefaultFlags) {
    CUevent event_;
    result = cuEventCreate(&event_, CU_EVENT_DEFAULT);

    EXPECT_EQ(result, CUDA_SUCCESS);

    cuEventDestroy(event_);
}

TEST_F(CuEventTest, AC_BA_EventCreate_BlockingSync) {
    CUevent event_;
    result = cuEventCreate(&event_, CU_EVENT_BLOCKING_SYNC);

    EXPECT_EQ(result, CUDA_SUCCESS);

    result = cuEventRecord(event_, 0);
    EXPECT_EQ(result, CUDA_SUCCESS);

    result = cuEventSynchronize(event_);
    EXPECT_EQ(result, CUDA_SUCCESS);

    result = cuEventQuery(event_);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuEventDestroy(event_);
}

TEST_F(CuEventTest, AC_BA_EventCreate_DisableTiming) {
    CUevent event_;
    result = cuEventCreate(&event_, CU_EVENT_DISABLE_TIMING);

    EXPECT_EQ(result, CUDA_SUCCESS);

    float time;
    result = cuEventElapsedTime(&time, event_, event_);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);

    cuEventDestroy(event_);
}

TEST_F(CuEventTest, AC_BA_EventCreate_CreateMultiple) {
    // create multiple events
    const int num_events = 10;
    CUevent events[num_events];
    for (int i = 0; i < num_events; i++)
    {
        cuEventCreate(&events[i], CU_EVENT_DEFAULT);
    }

    for (int i = 0; i < num_events; i++)
    {
        EXPECT_NE((CUevent) nullptr, events[i]);
    }

    for (int i = 0; i < num_events; i++)
    {
        cuEventDestroy(events[i]);
    }
}
