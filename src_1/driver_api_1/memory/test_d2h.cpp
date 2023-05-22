#include "cuda.h"
#include "gtest/gtest.h"

class CudaMemcpyTest : public ::testing::Test {
protected:
    CUdeviceptr dptr;
    void* hptr;
    size_t size;
    int init_value;

    void SetUp() override {
        size = 1024;
        init_value = 123;

        // Allocate host memory
        cuMemAllocHost(&hptr, size);

        // Allocate device memory
        cuMemAlloc(&dptr, size);

        // Initialize device memory
        cuMemsetD32(dptr, init_value, size / sizeof(int));
    }

    void TearDown() override {
        cuMemFree(dptr);
        cuMemFreeHost(hptr);
    }
};

TEST_F(CudaMemcpyTest, TestCuMemcpyDtoHSuccess) {
    CUresult result = cuMemcpyDtoH(hptr, dptr, size);
    EXPECT_EQ(result, CUDA_SUCCESS);

    // Check that the data was correctly copied
    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_EQ(static_cast<int*>(hptr)[i], init_value);
    }
}

TEST_F(CudaMemcpyTest, TestCuMemcpyDtoHZeroBytes) {
    CUresult result = cuMemcpyDtoH(hptr, dptr, 0);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyTest, TestCuMemcpyDtoHExcessiveSize) {
    CUresult result = cuMemcpyDtoH(hptr, dptr, static_cast<size_t>(-1));
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyTest, TestCuMemcpyDtoHNullDst) {
    CUresult result = cuMemcpyDtoH(nullptr, dptr, size);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyTest, TestCuMemcpyDtoHNullSrc) {
    CUresult result = cuMemcpyDtoH(hptr, 0, size);
    EXPECT_NE(result, CUDA_SUCCESS);
}
