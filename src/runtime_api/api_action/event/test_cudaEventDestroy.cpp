#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <math.h>
#include "test_case_with_kernel.h"

class CudaEventTest : public ::testing::Test {
   protected:
    cudaEvent_t event;

    void SetUp() override { 
        // EXPECT_EQ(cudaEventCreate(&event), cudaSuccess); 
    }

    void TearDown() override {
        // EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
    }

    static void destroyEvent(cudaEvent_t* event) {
        cudaEventDestroy(*event);
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
};

TEST_F(CudaEventTest, EventDestroy) {
    cudaEvent_t event;
    cudaEventCreate(&event);
    EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
    cudaEventDestroy(event);
}

TEST_F(CudaEventTest, DestroyConsumedEvent) {
    GTEST_SKIP();
    cudaEvent_t event;
    cudaEventCreate(&event);
    std::thread destroyThread(&CudaEventTest::destroyEvent, &event);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    cudaError_t result = cudaEventSynchronize(event);

    if (result != cudaSuccess) {
        const char *errorName, *errorString;
        errorName = cudaGetErrorName(result);
        errorString = cudaGetErrorString(result);
        FAIL() << "Error during event synchronization: " << errorName << ": "
               << errorString;
    }

    destroyThread.join();
}

TEST_F(CudaEventTest, DeviceSwitch) {
    cudaEvent_t event;
    cudaEventCreate(&event);
    int device;
    EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
    std::cout << device << std::endl;
    int count;
    cudaGetDeviceCount(&count);
    std::cout << count << std::endl;
    cudaEvent_t event2;
    EXPECT_EQ(cudaEventCreate(&event2), cudaSuccess);
    EXPECT_EQ(cudaSetDevice(device + 1), cudaSuccess);
    EXPECT_EQ(cudaEventRecord(event2, 0), cudaSuccess);
    EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(event2), cudaSuccess);
    cudaEventDestroy(event);
}

TEST_F(CudaEventTest, DestroyEventInUse) {
    cudaEvent_t event_;
    EXPECT_EQ(cudaEventCreate(&event_), cudaSuccess);

    EXPECT_EQ(cudaEventRecord(event_, 0), cudaSuccess);
    sleep(1);

    EXPECT_EQ(cudaEventDestroy(event_), cudaErrorNotReady);

    EXPECT_EQ(cudaEventSynchronize(event_), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(event_), cudaSuccess);
}

TEST_F(CudaEventTest, DestroyEventUsedInWait) {
    cudaEvent_t event;
    cudaEventCreate(&event, 0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamWaitEvent(stream, event, 0);

    // attempt to destroy the event, should return an error
    EXPECT_EQ(cudaEventDestroy(event), cudaErrorNotReady);

    cudaStreamDestroy(stream);
    cudaEventDestroy(event);
}

TEST_F(CudaEventTest, DestroyEventTwice) {
    cudaEvent_t event;
    EXPECT_EQ(cudaEventCreate(&event), cudaSuccess);

    // Destroy the event twice
    EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(event), cudaErrorInvalidResourceHandle);
}

TEST_F(CudaEventTest, DestroyInvalidEvent) {
    cudaEvent_t event = nullptr;
    EXPECT_EQ(cudaEventDestroy(event), cudaErrorInvalidResourceHandle);
}

TEST_F(CudaEventTest, multithreadedEventDestruction) {
    // 有几率出现core dump
    GTEST_SKIP();
    multithreadedEventDestruction();
}

TEST_F(CudaEventTest, DestroyByAnotherStream) {
    event_other_thread_use();
}