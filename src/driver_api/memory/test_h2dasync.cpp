#include <cuda.h>
#include <gtest/gtest.h>

// A helper macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        CUresult error = call; \
        ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
    } while (0)

// A test fixture class for cuMemcpyHtoDAsync tests
class CuMemcpyHtoDAsyncTest : public ::testing::Test {
protected:
    // Set up the test environment
    void SetUp() override {
        // Initialize the CUDA driver API
        CUDA_CHECK(cuInit(0));
        // Get the first device
        CUDA_CHECK(cuDeviceGet(&device, 0));
        // Create a context
        CUDA_CHECK(cuCtxCreate(&context, 0, device));
        // Allocate some host memory
        CUDA_CHECK(cuMemAllocHost(&hostPtr, size));
        // Allocate some device memory
        CUDA_CHECK(cuMemAlloc(&devicePtr, size));
    }

    // Tear down the test environment
    void TearDown() override {
        // Free the host memory
        CUDA_CHECK(cuMemFreeHost(hostPtr));
        // Free the device memory
        CUDA_CHECK(cuMemFree(devicePtr));
        // Destroy the context
        CUDA_CHECK(cuCtxDestroy(context));
    }

    CUdevice device; // The device handle
    CUcontext context; // The context handle
    CUdeviceptr devicePtr; // The device memory pointer
    void* hostPtr; // The host memory pointer
    const size_t size = 1024; // The size of the memory in bytes
};

// A test case for basic api behavior
TEST_F(CuMemcpyHtoDAsyncTest, BasicApiBehavior) {
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    // Initialize the host memory with some data
    for (size_t i = 0; i < size; i++) {
        ((char*)hostPtr)[i] = i % 256;
    }
    // Copy the host memory to the device memory asynchronously
    CUDA_CHECK(cuMemcpyHtoDAsync(devicePtr, hostPtr, size, stream));
    // Wait for the copy to finish
    CUDA_CHECK(cuStreamSynchronize(stream));
    // Verify the device memory content
    for (size_t i = 0; i < size; i++) {
        char value;
        CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
        EXPECT_EQ(value, i % 256);
    }
    // Destroy the stream
    CUDA_CHECK(cuStreamDestroy(stream));
}

// A test case for invalid arguments
TEST_F(CuMemcpyHtoDAsyncTest, InvalidArguments) {
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    // Try to copy with a null host pointer
    CUresult error = cuMemcpyHtoDAsync(devicePtr, nullptr, size, stream);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a null device pointer
    error = cuMemcpyHtoDAsync(0, hostPtr, size, stream);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a negative size
    error = cuMemcpyHtoDAsync(devicePtr, hostPtr, -1, stream);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a size larger than the allocated memory
    error = cuMemcpyHtoDAsync(devicePtr, hostPtr, size + 1, stream);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a null stream
    error = cuMemcpyHtoDAsync(devicePtr, hostPtr, size, nullptr);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Destroy the stream
    CUDA_CHECK(cuStreamDestroy(stream));
}

// A test case for boundary values
TEST_F(CuMemcpyHtoDAsyncTest, BoundaryValues) {
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    // Try to copy zero bytes
    CUresult error = cuMemcpyHtoDAsync(devicePtr, hostPtr, 0, stream);
    EXPECT_EQ(error, CUDA_SUCCESS);
    // Try to copy the maximum possible bytes
    size_t freeMem, totalMem;
    CUDA_CHECK(cuMemGetInfo(&freeMem, &totalMem));
    void* bigHostPtr;
    CUdeviceptr bigDevicePtr;
    CUDA_CHECK(cuMemAllocHost(&bigHostPtr, freeMem));
    CUDA_CHECK(cuMemAlloc(&bigDevicePtr, freeMem));
    error = cuMemcpyHtoDAsync(bigDevicePtr, bigHostPtr, freeMem, stream);
    EXPECT_EQ(error, CUDA_SUCCESS);
    CUDA_CHECK(cuMemFreeHost(bigHostPtr));
    CUDA_CHECK(cuMemFree(bigDevicePtr));
    // Try to copy unaligned addresses
        error = cuMemcpyHtoDAsync(devicePtr + 1, hostPtr + 1, size - 2, stream);
    EXPECT_EQ(error, CUDA_SUCCESS);
    // Destroy the stream
    CUDA_CHECK(cuStreamDestroy(stream));
}

// A test case for synchronous and asynchronous behavior
TEST_F(CuMemcpyHtoDAsyncTest, SyncAsyncBehavior) {
    // Create two streams
    CUstream stream1, stream2;
    CUDA_CHECK(cuStreamCreate(&stream1, 0));
    CUDA_CHECK(cuStreamCreate(&stream2, 0));
    // Initialize the host memory with some data
    for (size_t i = 0; i < size; i++) {
        ((char*)hostPtr)[i] = i % 256;
    }
    // Copy the host memory to the device memory asynchronously in stream1
    CUDA_CHECK(cuMemcpyHtoDAsync(devicePtr, hostPtr, size, stream1));
    // Copy the device memory to the host memory asynchronously in stream2
    CUDA_CHECK(cuMemcpyDtoHAsync(hostPtr, devicePtr, size, stream2));
    // Check if the copies are completed
    CUresult error1 = cuStreamQuery(stream1);
    CUresult error2 = cuStreamQuery(stream2);
    EXPECT_TRUE(error1 == CUDA_SUCCESS || error1 == CUDA_ERROR_NOT_READY);
    EXPECT_TRUE(error2 == CUDA_SUCCESS || error2 == CUDA_ERROR_NOT_READY);
    // Wait for the copies to finish
    CUDA_CHECK(cuStreamSynchronize(stream1));
    CUDA_CHECK(cuStreamSynchronize(stream2));
    // Check if the copies are completed
    error1 = cuStreamQuery(stream1);
    error2 = cuStreamQuery(stream2);
    EXPECT_EQ(error1, CUDA_SUCCESS);
    EXPECT_EQ(error2, CUDA_SUCCESS);
    // Verify the host memory content
    for (size_t i = 0; i < size; i++) {
        char value = ((char*)hostPtr)[i];
        EXPECT_EQ(value, i % 256);
    }
    // Destroy the streams
    CUDA_CHECK(cuStreamDestroy(stream1));
    CUDA_CHECK(cuStreamDestroy(stream2));
}

// A test case for other testing contents
TEST_F(CuMemcpyHtoDAsyncTest, OtherTestingContents) {
    // Get the number of devices
    int deviceCount;
    CUDA_CHECK(cuDeviceGetCount(&deviceCount));
    // Loop over all devices
    for (int i = 0; i < deviceCount; i++) {
        // Get the device handle
        CUdevice device;
        CUDA_CHECK(cuDeviceGet(&device, i));
        // Create a context
        CUcontext context;
        CUDA_CHECK(cuCtxCreate(&context, 0, device));
        // Push the context to the current thread
        CUDA_CHECK(cuCtxPushCurrent(context));
        // Allocate some device memory
        CUdeviceptr devicePtr;
        CUDA_CHECK(cuMemAlloc(&devicePtr, size));
        // Create a stream
        CUstream stream;
        CUDA_CHECK(cuStreamCreate(&stream, 0));
        // Copy the host memory to the device memory asynchronously
        CUDA_CHECK(cuMemcpyHtoDAsync(devicePtr, hostPtr, size, stream));
        // Wait for the copy to finish
        CUDA_CHECK(cuStreamSynchronize(stream));
        // Verify the device memory content
        for (size_t i = 0; i < size; i++) {
            char value;
            CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
            EXPECT_EQ(value, i % 256);
        }
        // Destroy the stream
        CUDA_CHECK(cuStreamDestroy(stream));
        // Free the device memory
        CUDA_CHECK(cuMemFree(devicePtr));
        // Pop the context from the current thread
        CUDA_CHECK(cuCtxPopCurrent(&context));
        // Destroy the context
        CUDA_CHECK(cuCtxDestroy(context));
    }
}


// A test case for loop testing
TEST_F(CuMemcpyHtoDAsyncTest, LoopTesting) {
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    // Initialize the host memory with some data
    for (size_t i = 0; i < size; i++) {
        ((char*)hostPtr)[i] = i % 256;
    }
    // Copy the host memory to the device memory asynchronously in a loop
    const int loopCount = 10;
    for (int j = 0; j < loopCount; j++) {
        CUDA_CHECK(cuMemcpyHtoDAsync(devicePtr, hostPtr, size, stream));
    }
    // Wait for the copies to finish
    CUDA_CHECK(cuStreamSynchronize(stream));
    // Verify the device memory content
    for (size_t i = 0; i < size; i++) {
        char value;
        CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
        EXPECT_EQ(value, i % 256);
    }
    // Destroy the stream
    CUDA_CHECK(cuStreamDestroy(stream));
}
