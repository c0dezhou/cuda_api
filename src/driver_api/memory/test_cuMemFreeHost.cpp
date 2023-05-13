#include "cuda.h"
#include "gtest/gtest.h"

class CudaFreeTest : public ::testing::Test {
protected:
    void* ptr;
    size_t size;

    void SetUp() override {
        // Assuming size is set as per requirements for the tests.
        size = 1024;
        ptr = nullptr;
        cuMemAllocHost(&ptr, size);
    }

    void TearDown() override {
        // No freeing in teardown because we want to test this in the tests
    }
};

TEST_F(CudaFreeTest, TestCuMemFreeHostSuccess) {
    CUresult result = cuMemFreeHost(ptr);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CudaFreeTest, TestCuMemFreeHostNullPointer) {
    cuMemFreeHost(ptr);  // Free the memory allocated in SetUp
    ptr = nullptr;
    CUresult result = cuMemFreeHost(ptr);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaFreeTest, TestCuMemFreeHostDoubleFree) {
    CUresult result = cuMemFreeHost(ptr);
    EXPECT_EQ(result, CUDA_SUCCESS);
    result = cuMemFreeHost(ptr);  // Attempt to free the same memory again
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaFreeTest, TestCuMemFreeHostSyncBehavior) {
    CUresult result = cuMemFreeHost(ptr);
    EXPECT_EQ(result, CUDA_SUCCESS);
    
    // We should not be able to access the memory after cuMemFreeHost,
    // as cuMemFreeHost is synchronous and blocks until operation completes.
    EXPECT_DEATH_IF_SUPPORTED(*reinterpret_cast<int*>(ptr), "");
}