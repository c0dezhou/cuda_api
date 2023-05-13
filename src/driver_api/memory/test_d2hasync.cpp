#include "cuda.h"
#include "gtest/gtest.h"

class CudaMemcpyAsyncTest : public ::testing::Test {
protected:
    CUdeviceptr dptr;
    void* hptr;
    size_t size;
    int init_value;
    CUstream stream;

    void SetUp() override {
        size = 1024;
        init_value = 123;

        // Allocate host memory
        cuMemAllocHost(&hptr, size);

        // Allocate device memory
        cuMemAlloc(&dptr, size);

        // Initialize device memory
        cuMemsetD32(dptr, init_value, size / sizeof(int));

        // Create a stream
        cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    }

    void TearDown() override {
        cuStreamDestroy(stream);
        cuMemFree(dptr);
        cuMemFreeHost(hptr);
    }
};

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncSuccess) {
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, size, stream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    // Wait for the copy operation to complete
    cuStreamSynchronize(stream);

    // Check that the data was correctly copied
    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_EQ(static_cast<int*>(hptr)[i], init_value);
    }
}

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncZeroBytes) {
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, 0, stream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    // Even though we're copying zero bytes, we should still synchronize
    cuStreamSynchronize(stream);
}

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncExcessiveSize) {
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, static_cast<size_t>(-1), stream);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncNullDst) {
    CUresult result = cuMemcpyDtoHAsync(nullptr, dptr, size, stream);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncNullSrc) {
    CUresult result = cuMemcpyDtoHAsync(hptr, 0, size, stream);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(CudaMemcpyAsyncTest, TestCuMemcpyDtoHAsyncInvalidStream) {
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, size, nullptr);
    EXPECT_NE(result, CUDA_SUCCESS);
}
