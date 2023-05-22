#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <math.h>

__global__ void dummyKernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        float result = 0.0f;
        for (int i = 0; i < 1000000; ++i) {
            result += sinf(static_cast<float>(i)) * cosf(static_cast<float>(i));
        }
    }
}

void createAndDestroyEvent(cudaEvent_t* event, std::atomic<bool>& done) {
    cudaEventCreate(event);
    dummyKernel<<<1, 1>>>();
    cudaEventRecord(*event, 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cudaEventDestroy(*event);
    done.store(true);
}

void waitForEvent(cudaEvent_t* event,
                  std::atomic<bool>& done,
                  cudaError_t* result) {
    while (!done.load()) {
        *result = cudaEventSynchronize(*event);
        if (*result != cudaSuccess) {
            break;
        }
    }
}

void multithreadedEventDestruction() {
    std::atomic<bool> done(false);
    cudaEvent_t event;
    cudaError_t waitForEventResult;

    // Create two threads, one for creating and destroying events, and another
    // for waiting on events
    std::thread createAndDestroyThread(createAndDestroyEvent, &event,
                                       std::ref(done));
    std::thread waitForEventThread(waitForEvent, &event, std::ref(done),
                                   &waitForEventResult);

    createAndDestroyThread.join();
    waitForEventThread.join();


    // Check if cudaEventSynchronize returned an error
    EXPECT_NE(waitForEventResult, cudaSuccess);
}

void event_other_thread_use() {
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    dummyKernel<<<1, 1, 0, stream1>>>();
    cudaEventRecord(event1, stream1);
    dummyKernel<<<1, 1, 0, stream2>>>();
    cudaEventRecord(event2, stream2);

    // Add cuStreamWaitEvent to make stream2 wait for event1
    cudaStreamWaitEvent(stream2, event1, 0);

    // Wait until both streams are actively executing the kernels
    while (cudaStreamQuery(stream2) == cudaErrorNotReady) {

        cudaError_t result1 = cudaEventDestroy(event1);
        cudaError_t result2 = cudaEventDestroy(event2);

        EXPECT_NE(result1, cudaSuccess);
        EXPECT_NE(result2, cudaSuccess);
    }

    

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

}