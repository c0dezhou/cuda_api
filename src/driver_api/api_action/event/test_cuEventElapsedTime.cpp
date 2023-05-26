#include "event_tests.h"

TEST_F(CuEventTest, AC_BA_EventElapsedTime_CalculateElapsedTime) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(start_, stream);

    sleep(3);

    cuEventRecord(end_, stream);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    float elapsed_time;
    result = cuEventElapsedTime(&elapsed_time, start_, end_);
    EXPECT_EQ(CUDA_SUCCESS, result);

    EXPECT_NEAR(elapsed_time, 3000.0f, 3.0f);
}

TEST_F(CuEventTest, AC_BA_EventElapsedTime_CalculateElapsedTime2stream) {
    CUstream stream1, stream2;
    cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING);

    CUdeviceptr d_src, d_dst1, d_dst2;
    size_t dataSize = 1024 * sizeof(float);
    cuMemAlloc(&d_src, dataSize);
    cuMemAlloc(&d_dst1, dataSize);
    cuMemAlloc(&d_dst2, dataSize);

    cuEventRecord(start_, stream1);
    cuMemcpyAsync(d_dst1, d_src, dataSize, stream1);
    cuStreamSynchronize(stream1);
    cuMemcpyAsync(d_dst2, d_src, dataSize, stream1);
    cuStreamSynchronize(stream1);
    cuEventRecord(end_, stream1);
    cuEventSynchronize(end_);
    float elapsedTimeSequential = calculateElapsedTime(start_, end_);

    cuEventRecord(start_, stream1);
    cuMemcpyAsync(d_dst1, d_src, dataSize, stream1);
    cuMemcpyAsync(d_dst2, d_src, dataSize, stream2);
    cuStreamSynchronize(stream1);
    cuStreamSynchronize(stream2);
    cuEventRecord(end_, stream1);
    cuEventSynchronize(end_);
    float elapsedTimeParallel = calculateElapsedTime(start_, end_);

    std::cout << elapsedTimeSequential << " " << elapsedTimeParallel
              << std::endl;

    EXPECT_GT(elapsedTimeSequential, 0.0f);
    EXPECT_GT(elapsedTimeParallel, 0.0f);

    EXPECT_GT(elapsedTimeSequential, elapsedTimeParallel);

    cuMemFree(d_src);
    cuMemFree(d_dst1);
    cuMemFree(d_dst2);

    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
}

TEST_F(CuEventTest, AC_INV_EventElapsedTime_CalculateElapsedTimeNotRecode) {
    float elapsed_time;
    result = cuEventElapsedTime(&elapsed_time, start_, end_);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);
}

TEST_F(CuEventTest, AC_INV_EventElapsedTime_CalculateElapsedTimeNullPointer) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(start_, stream);
    cuEventRecord(end_, stream);
    cuStreamSynchronize(stream);

    result = cuEventElapsedTime(nullptr, start_, end_);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, result);

    cuStreamDestroy(stream);
}

TEST_F(CuEventTest, AC_INV_EventElapsedTime_CalculateElapsedTimeSameEvent) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    cuEventRecord(start_, stream);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    float elapsed_time;
    result = cuEventElapsedTime(&elapsed_time, start_, start_);
    EXPECT_EQ(CUDA_SUCCESS, result);
    EXPECT_EQ(0.0f, elapsed_time);
}

TEST_F(CuEventTest, AC_BA_EventElapsedTime_CompareToHostRecord) {
    // TODO: 解决
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    cuEventRecord(start_, stream);
    cuStreamSynchronize(stream);  // Ensure start_ event has completed

    const int num_iterations = 100000000;
    auto ss = std::chrono::steady_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        volatile int x = i * i + 1;
    }
    auto ee = std::chrono::steady_clock::now();

    cuEventRecord(end_, stream);
    cuStreamSynchronize(stream);  // Ensure end_ event has completed

    float hosttime =
        std::chrono::duration_cast<std::chrono::milliseconds>(ee - ss).count();

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    float elapsed_time_ms;
    result = cuEventElapsedTime(&elapsed_time_ms, start_, end_);

    EXPECT_EQ(CUDA_SUCCESS, result);

    const float expected_time_ms = 1.0f;
    const float tolerance_ms = 0.2f;
    EXPECT_NEAR(hosttime, elapsed_time_ms, tolerance_ms);

    cuEventDestroy(start_);
    cuEventDestroy(end_);
}
