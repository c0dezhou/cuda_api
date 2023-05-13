#include <cuda.h>
#include <gtest/gtest.h>

// A helper macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        CUresult error = call; \
        ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
    } while (0)

// A test fixture class for cuMemcpyHtoD tests
class CuMemcpyHtoDTest : public ::testing::Test {
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
TEST_F(CuMemcpyHtoDTest, BasicApiBehavior) {
    // Initialize the host memory with some data
    for (size_t i = 0; i < size; i++) {
        ((char*)hostPtr)[i] = i % 256;
    }
    // Copy the host memory to the device memory
    CUDA_CHECK(cuMemcpyHtoD(devicePtr, hostPtr, size));
    // Verify the device memory content
    for (size_t i = 0; i < size; i++) {
        char value;
        CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
        EXPECT_EQ(value, i % 256);
    }
}

// A test case for invalid arguments
TEST_F(CuMemcpyHtoDTest, InvalidArguments) {
    // Try to copy with a null host pointer
    CUresult error = cuMemcpyHtoD(devicePtr, nullptr, size);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a null device pointer
    error = cuMemcpyHtoD(0, hostPtr, size);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a negative size
    error = cuMemcpyHtoD(devicePtr, hostPtr, -1);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to copy with a size larger than the allocated memory
    error = cuMemcpyHtoD(devicePtr, hostPtr, size + 1);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
}

// A test case for boundary values
TEST_F(CuMemcpyHtoDTest, BoundaryValues) {
    // Try to copy zero bytes
    CUresult error = cuMemcpyHtoD(devicePtr, hostPtr, 0);
    EXPECT_EQ(error, CUDA_SUCCESS);
    // Try to copy the maximum possible bytes
    size_t freeMem, totalMem;
    CUDA_CHECK(cuMemGetInfo(&freeMem, &totalMem));
    void* bigHostPtr;
    CUdeviceptr bigDevicePtr;
    CUDA_CHECK(cuMemAllocHost(&bigHostPtr, freeMem));
    CUDA_CHECK(cuMemAlloc(&bigDevicePtr, freeMem));
    error = cuMemcpyHtoD(bigDevicePtr, bigHostPtr, freeMem);
    EXPECT_EQ(error, CUDA_SUCCESS);
    CUDA_CHECK(cuMemFreeHost(bigHostPtr));
    CUDA_CHECK(cuMemFree(bigDevicePtr));
    // Try to copy unaligned addresses
    error = cuMemcpyHtoD(devicePtr + 1, hostPtr + 1, size - 2);
    EXPECT_EQ(error, CUDA_SUCCESS);
}

// A test case for synchronous and asynchronous behavior
TEST_F(CuMemcpyHtoDTest, SyncAsyncBehavior) {
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    // Copy the host memory to the device memory asynchronously
    CUDA_CHECK(cuMemcpyHtoDAsync(devicePtr, hostPtr, size, stream));
    // Check if the copy is completed
    CUresult error = cuStreamQuery(stream);
    EXPECT_TRUE(error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY);
    // Wait for the copy to finish
    CUDA_CHECK(cuStreamSynchronize(stream));
    // Check if the copy is completed
    error = cuStreamQuery(stream);
    EXPECT_EQ(error, CUDA_SUCCESS);
    // Destroy the stream
    CUDA_CHECK(cuStreamDestroy(stream));
}

// A test case for other testing contents
TEST_F(CuMemcpyHtoDTest, OtherTestingContents) {
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
        // Copy the host memory to the device memory
        CUDA_CHECK(cuMemcpyHtoD(devicePtr, hostPtr, size));
        // Verify the device memory content
        for (size_t i = 0; i < size; i++) {
            char value;
            CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
            EXPECT_EQ(value, i % 256);
        }
        // Free the device memory
        CUDA_CHECK(cuMemFree(devicePtr));
        // Pop the context from the current thread
        CUDA_CHECK(cuCtxPopCurrent(&context));
        // Destroy the context
        CUDA_CHECK(cuCtxDestroy(context));
    }
}