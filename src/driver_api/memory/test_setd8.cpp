#include <cuda.h>
#include <gtest/gtest.h>

// A helper macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        CUresult error = call; \
        ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
    } while (0)

// A test fixture class for cuMemsetD8 tests
class CuMemsetD8Test : public ::testing::Test {
protected:
    // Set up the test environment
    void SetUp() override {
        // Initialize the CUDA driver API
        CUDA_CHECK(cuInit(0));
        // Get the first device
        CUDA_CHECK(cuDeviceGet(&device, 0));
        // Create a context
        CUDA_CHECK(cuCtxCreate(&context, 0, device));
        // Allocate some device memory
        CUDA_CHECK(cuMemAlloc(&devicePtr, size));
    }

    // Tear down the test environment
    void TearDown() override {
        // Free the device memory
        CUDA_CHECK(cuMemFree(devicePtr));
        // Destroy the context
        CUDA_CHECK(cuCtxDestroy(context));
    }

    CUdevice device; // The device handle
    CUcontext context; // The context handle
    CUdeviceptr devicePtr; // The device memory pointer
    const size_t size = 1024; // The size of the memory in bytes
};

// A test case for basic api behavior
TEST_F(CuMemsetD8Test, BasicApiBehavior) {
    // Set the device memory to a value
    unsigned char value = 42;
    CUDA_CHECK(cuMemsetD8(devicePtr, value, size));
    // Verify the device memory content
        for (size_t i = 0; i < size; i++) {
        unsigned char value;
        CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
        EXPECT_EQ(value, 42);
    }
}

// A test case for invalid arguments
TEST_F(CuMemsetD8Test, InvalidArguments) {
    // Try to set with a null device pointer
    CUresult error = cuMemsetD8(0, 42, size);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to set with a negative size
    error = cuMemsetD8(devicePtr, 42, -1);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
    // Try to set with a size larger than the allocated memory
    error = cuMemsetD8(devicePtr, 42, size + 1);
    EXPECT_EQ(error, CUDA_ERROR_INVALID_VALUE);
}

// A test case for boundary values
TEST_F(CuMemsetD8Test, BoundaryValues) {
    // Try to set zero bytes
    CUresult error = cuMemsetD8(devicePtr, 42, 0);
    EXPECT_EQ(error, CUDA_SUCCESS);
    // Try to set the maximum possible bytes
    size_t freeMem, totalMem;
    CUDA_CHECK(cuMemGetInfo(&freeMem, &totalMem));
    CUdeviceptr bigDevicePtr;
    CUDA_CHECK(cuMemAlloc(&bigDevicePtr, freeMem));
    error = cuMemsetD8(bigDevicePtr, 42, freeMem);
    EXPECT_EQ(error, CUDA_SUCCESS);
    CUDA_CHECK(cuMemFree(bigDevicePtr));
    // Try to set unaligned addresses
    error = cuMemsetD8(devicePtr + 1, 42, size - 2);
    EXPECT_EQ(error, CUDA_SUCCESS);
}

// A test case for other testing contents
TEST_F(CuMemsetD8Test, OtherTestingContents) {
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
        // Set the device memory to a value
        unsigned char value = 42;
        CUDA_CHECK(cuMemsetD8(devicePtr, value, size));
        // Verify the device memory content
        for (size_t i = 0; i < size; i++) {
            unsigned char value;
            CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
            EXPECT_EQ(value, 42);
        }
        // Free the device memory
        CUDA_CHECK(cuMemFree(devicePtr));
        // Pop the context from the current thread
        CUDA_CHECK(cuCtxPopCurrent(&context));
        // Destroy the context
        CUDA_CHECK(cuCtxDestroy(context));
    }
}

// A test case for loop testing
TEST_F(CuMemsetD8Test, LoopTesting) {
    // Set the device memory to a value in a loop
    unsigned char value = 42;
    const int loopCount = 10;
    for (int j = 0; j < loopCount; j++) {
        CUDA_CHECK(cuMemsetD8(devicePtr, value, size));
    }
    // Verify the device memory content
        for (size_t i = 0; i < size; i++) {
        unsigned char value;
        CUDA_CHECK(cuMemcpyDtoH(&value, devicePtr + i, 1));
        EXPECT_EQ(value, 42);
    }
}
