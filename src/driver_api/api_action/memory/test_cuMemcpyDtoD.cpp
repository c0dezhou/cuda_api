#include "memory_tests.h"
#include "cuda_runtime.h"

#define INIT_MEMD2D()                \
    CUdeviceptr d_src_;              \
    CUdeviceptr d_dst_;              \
    int* h_ptr_;                     \
    static const size_t N = 5;     \
    size_t size = N * sizeof(int);   \
    cuMemAlloc(&d_src_, size);       \
    cuMemAlloc(&d_dst_, size);       \
    h_ptr_ = new int[N];             \
    for (size_t i = 0; i < N; i++) { \
        h_ptr_[i] = i;               \
    }                                \
    cuMemcpyHtoD(d_src_, h_ptr_, size);

#define DEL_MEMD2D()   \
    delete[] h_ptr_;   \
    cuMemFree(d_src_); \
    cuMemFree(d_dst_);

TEST_F(CuMemTest, AC_BA_MemcpyDtoD_BasicBehavior) {
    INIT_MEMD2D();
    cuMemcpyDtoD(d_dst_, d_src_, size);

    cudaError_t res = cudaDeviceSynchronize();
    EXPECT_EQ(res, 0);

    cuMemcpyDtoH(h_ptr_, d_dst_, size);

    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_ptr_[i], i);
    }
    DEL_MEMD2D();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoD_InvalidArguments) {
    INIT_MEMD2D();
    CUdeviceptr d_src_invalid = NULL;
    CUdeviceptr d_dst_invalid = NULL;

    size_t size_invalid = (N + 1) * sizeof(int);

    CUresult res = cuMemcpyDtoD(d_dst_invalid, d_src_, N * sizeof(int));
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    res = cuMemcpyDtoD(d_dst_, d_src_invalid, N);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    DEL_MEMD2D();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoD_ZeroByte) {
    INIT_MEMD2D();
    CUresult res = cuMemcpyDtoD(d_dst_, d_src_, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMD2D();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoD_HoleDeviceMem) {
    INIT_MEMD2D();
    size_t size_0 = 0;
    cuMemGetInfo(NULL, &size_0);
    res = cuMemcpyDtoD(d_dst_, d_src_, size);
    EXPECT_EQ(res, CUDA_SUCCESS);
    DEL_MEMD2D();
}

TEST_F(CuMemTest, AC_OT_MemcpyDtoD_OverlapMem) {
    //TODO: 解决 
    // 正在尝试执行源和目标范围重叠的设备到设备内存复制操作。 CUDA 文档指出：“使用不满足这些条件的 srcDevice 和 dstDevice 指针调用 cuMemcpyDtoD() 会导致未定义的行为。”
    CUdeviceptr d_src_;              
    CUdeviceptr d_dst_;             
    int* h_ptr_;                     
    static const size_t N = 5;     
    size_t size = N * sizeof(int);   
    cuMemAlloc(&d_src_, size);       
    cuMemAlloc(&d_dst_, size);       
    h_ptr_ = new int[N];             
    for (size_t i = 0; i < N; i++) { 
        h_ptr_[i] = i;               
    }                                
    cuMemcpyHtoD(d_src_, h_ptr_, size);
    
    // The source and destination ranges for cuMemcpyDtoD should not overlap
    // We'll just copy to the beginning of d_dst_
    res = cuMemcpyDtoD(d_dst_, d_src_, size / 2);
    EXPECT_EQ(res, CUDA_SUCCESS);

    cudaDeviceSynchronize();

    cuMemcpyDtoH(h_ptr_, d_dst_, size);

    for (size_t i = 0; i < N / 2; i++) {
        EXPECT_EQ(h_ptr_[i], i); // These should match the original data
    }
    for (size_t i = N / 2; i < N; i++) {
        EXPECT_EQ(h_ptr_[i], 0); // These have not been written to, should be zero
    }

    delete[] h_ptr_;   
    cuMemFree(d_src_); 
    cuMemFree(d_dst_);
}


TEST_F(CuMemTest, AC_SA_MemcpyDtoD_SyncBehavior) {
    INIT_MEMD2D();
    CUstream hStream;
    cuStreamCreate(&hStream, CU_STREAM_DEFAULT);

    cuMemcpyDtoD(d_dst_, d_src_, size);

    // 此处不做同步，直接拷
    cuMemcpyDtoH(h_ptr_, d_dst_, size);

    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_ptr_[i], i); 
    }

    cuStreamDestroy(hStream);
    DEL_MEMD2D();
}
