#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_EventRecord_RecordedTimestamp) {
    CUevent startEvent, stopEvent;
    cuEventCreate(&startEvent, 0);
    cuEventCreate(&stopEvent, 0);

    cuEventRecord(startEvent, cuStream);

    int input_a = 5;
    int input_b = 7;
    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, cuStream, args, nullptr);
    cuMemcpyDtoHAsync(&result, (CUdeviceptr)d_result, sizeof(int), cuStream);

    cuEventRecord(stopEvent, cuStream);

    cuEventSynchronize(startEvent);
    cuEventSynchronize(stopEvent);

    float elapsedTime;
    cuEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    EXPECT_GE(elapsedTime, 0);

    cuMemFree((CUdeviceptr)d_result);
    cuEventDestroy(startEvent);
    cuEventDestroy(stopEvent);
    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_BA_EventRecord_MultipleRecordings) {
    CUevent event;
    cuEventCreate(&event, 0);

    cuEventRecord(event, cuStream);
    cuEventRecord(event, cuStream);

    cuEventSynchronize(event);

    cuEventDestroy(event);
    cuStreamDestroy(cuStream);
}

TEST_F(CuStreamTests, AC_SA_EventRecord_EventRecordingWithoutSynchronization) {
    // TODO
}
